import ray
from typing import List, Optional, Any
from .agent import Agent
from .journal import Node, Journal
from .interpreter import ExecutionResult, Interpreter
from .utils.config import Config
from omegaconf import OmegaConf
from .utils import data_preview as dp
import logging
from pathlib import Path

@ray.remote
class ParallelWorker(Agent):
    """Worker class that inherits from Agent to handle code generation and execution"""
    def __init__(self, task_desc: str, cfg: Config, journal: Journal, data_preview: str):
        super().__init__(task_desc, cfg, journal)
        # Initialize interpreter for this worker
        self.interpreter = Interpreter( 
            cfg.workspace_dir, **OmegaConf.to_container(cfg.exec)  # type: ignore
        )
        # Initialize data preview
        self.data_preview = data_preview
        # Setup logger for this worker
        actor_id = ray.get_runtime_context().get_actor_id()
        self.logger = logging.getLogger(f"ParallelWorker-{actor_id}")
        self.logger.setLevel(logging.INFO)

    def generate_nodes(self, parent_node: Optional[Node], num_nodes: int) -> List[Node]:
        """Generate multiple nodes in parallel"""
        self.logger.info(f"Generating {num_nodes} nodes from parent: {parent_node}")
        nodes = []
        for _ in range(num_nodes):
            if parent_node is None:
                node = self._draft()
            elif parent_node.is_buggy:
                node = self._debug(parent_node)
            else:
                node = self._improve(parent_node)
            nodes.append(node)
        self.logger.info(f"Generated {len(nodes)} nodes")
        return nodes

    def execute_and_evaluate_node(self, node: Node) -> Node:
        """Execute node's code and evaluate results"""
        try:
            actor_id = ray.get_runtime_context().get_actor_id()
            self.logger.info(f"Worker {actor_id} executing node {node.id}")
            # Execute code
            result = self.interpreter.run(node.code, True)
            # Absorb execution results
            node.absorb_exec_result(result)
            # Evaluate and update node metrics
            self.parse_exec_result(node, result)
            self.logger.info(f"Worker {actor_id} completed node {node.id} with metric: {node.metric.value if node.metric else 'None'}")
            return node
        except Exception as e:
            self.logger.error(f"Worker {actor_id} failed executing node {node.id}: {str(e)}")
            raise

    def get_data_preview(self):
        """Return the data preview"""
        return self.data_preview

    def cleanup_interpreter(self):
        """Cleanup the interpreter session"""
        self.interpreter.cleanup_session()

    def search_and_generate(self, num_nodes: int) -> List[Node]:
        """Independent search and generation by each worker"""
        parent_node = self.search_policy()
        self.logger.info(f"Worker selected parent node: {parent_node.id if parent_node else 'None'}")
        return self.generate_nodes(parent_node, num_nodes)

class ParallelAgent(Agent):
    """Main agent class that implements parallel tree search"""
    def __init__(self, task_desc: str, cfg: Config, journal: Journal):
        super().__init__(task_desc, cfg, journal)
        
        ray.init(
            ignore_reinit_error=True,
            logging_level=logging.INFO,
            )
        
        # Initialize data preview first
        if cfg.agent.data_preview:
            self.data_preview = dp.generate(cfg.workspace_dir)
        else:
            self.data_preview = None
            
        self.num_workers = cfg.agent.parallel.num_workers
        self.nodes_per_worker = cfg.agent.parallel.nodes_per_worker
        
        # Setup logger for parallel execution
        self.logger = logging.getLogger("aide.parallel")
        self.logger.setLevel(logging.INFO)
        
        self.workers = [
            ParallelWorker.remote(
                task_desc=task_desc, 
                cfg=cfg, 
                journal=journal,
                data_preview=self.data_preview
            ) 
            for _ in range(self.num_workers)
        ]

    def step(self, exec_callback: Any = None):
        """Single step of the tree search"""
        step_num = len(self.journal)
        self.logger.info(f"Starting step {step_num}")
        
        if not self.journal.nodes:
            self.update_data_preview()
            self.logger.info("Updated data preview")

        # Let workers independently search and generate nodes
        node_futures = [
            worker.search_and_generate.remote(self.nodes_per_worker)
            for worker in self.workers
        ]
        
        # Wait for node generation
        self.logger.info(f"Step {step_num}: Waiting for node generation to complete...")
        generated_nodes = ray.get(node_futures)
        total_nodes = sum(len(nodes) for nodes in generated_nodes)
        self.logger.info(f"Step {step_num}: Generated {total_nodes} nodes total")
        
        # Flatten list of nodes and maintain parent relationships
        all_nodes = []
        for worker_nodes in generated_nodes:
            for node in worker_nodes:
                all_nodes.append(node)
        
        # Distribute execution work across workers (same layer parallel execution)
        nodes_per_executor = max(1, len(all_nodes) // len(self.workers))
        exec_futures = []
        
        self.logger.info(f"Step {step_num}: Distributing {len(all_nodes)} nodes across {len(self.workers)} workers for execution")
        for i, worker in enumerate(self.workers):
            start_idx = i * nodes_per_executor
            end_idx = start_idx + nodes_per_executor if i < len(self.workers) - 1 else len(all_nodes)
            worker_nodes = all_nodes[start_idx:end_idx]
            
            self.logger.info(f"Step {step_num}: Worker {i} assigned {len(worker_nodes)} nodes")
            for node in worker_nodes:
                exec_futures.append(worker.execute_and_evaluate_node.remote(node))
        
        # Get execution results and update journal
        self.logger.info(f"Step {step_num}: Waiting for {len(exec_futures)} node executions to complete...")
        evaluated_nodes = ray.get(exec_futures)
        self.logger.info(f"Step {step_num}: All node executions completed")
        
        # Batch update journal and save results
        successful_nodes = 0
        buggy_nodes = 0
        best_metric = float('-inf')
        
        for node in evaluated_nodes:
            if node.parent is None:  # Check node's parent attribute instead of using parent_node
                self.journal.draft_nodes.append(node)
            self.journal.append(node)
            
            # Track statistics
            if node.is_buggy:
                buggy_nodes += 1
            else:
                successful_nodes += 1
                if node.metric and node.metric.value > best_metric:
                    best_metric = node.metric.value
        
        self.logger.info(
            f"Step {step_num} completed: "
            f"{successful_nodes} successful nodes, "
            f"{buggy_nodes} buggy nodes, "
            f"best metric: {best_metric if best_metric != float('-inf') else 'N/A'}"
        )
        
        # Save results
        try:
            from .utils.config import save_run
            save_run(self.cfg, self.journal)
            self.logger.info(f"Step {step_num}: Successfully saved run data to {self.cfg.log_dir}")
        except Exception as e:
            self.logger.error(f"Step {step_num}: Failed to save run: {str(e)}")

    def cleanup(self):
        """Cleanup Ray resources"""
        for worker in self.workers:
            ray.get(worker.cleanup_interpreter.remote())
        ray.shutdown()

    def update_data_preview(self):
        """Update data preview from the first worker"""
        if not hasattr(self, 'data_preview'):
            self.data_preview = ray.get(self.workers[0].get_data_preview.remote())
