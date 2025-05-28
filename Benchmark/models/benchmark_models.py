from typing import List, Dict, Any, Optional
from datetime import datetime

class TestCase:
    """Represents a single test case (question) loaded from a JSON file."""
    def __init__(self, technique_id: str, question_id: str, prompt: str, KQL_query: str, answer: List[str],
                 context: Optional[str] = None, objective: Optional[str] = None,
                 technical_details: Optional[Dict] = None,
                 thinking_how_to_phrase_question_and_answer: Optional[str] = None,
                 difficulty: Optional[str] = None, KQL_validation_message: Optional[str] = None,
                 KQL_query_results: Optional[List[List[Any]]] = None):
        self.technique_id = technique_id
        self.question_id = question_id
        self.prompt = prompt
        self.KQL_query = KQL_query
        self.answer = answer
        self.context = context
        self.objective = objective
        self.technical_details = technical_details
        self.thinking_how_to_phrase_question_and_answer = thinking_how_to_phrase_question_and_answer
        self.difficulty = difficulty
        self.KQL_validation_message = KQL_validation_message
        self.KQL_query_results = KQL_query_results

    def to_dict(self) -> Dict[str, Any]:
        # Simple conversion for this class, assuming all attributes are directly serializable
        # or handled by json.dumps (like List, Dict, str, int, etc.)
        return self.__dict__

class QueryResult:
    """Stores the results of a query execution"""
    def __init__(self, query: str, raw_results: List, answer: str, attempts: int,
                 execution_time: Optional[float] = None, cost: float = 0.0,
                 all_attempts: Optional[List[Dict]] = None, 
                 llm_formulate_kql_errors: int = 0):
        self.query = query
        self.raw_results = raw_results
        self.answer = answer
        self.attempts = attempts
        self.execution_time = execution_time
        self.cost = cost
        self.all_attempts = all_attempts if all_attempts is not None else []
        self.llm_formulate_kql_errors = llm_formulate_kql_errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "raw_results": self.raw_results,
            "answer": self.answer,
            "attempts": self.attempts,
            "execution_time": self.execution_time,
            "cost": self.cost,
            "all_attempts": self.all_attempts,
            "llm_formulate_kql_errors": self.llm_formulate_kql_errors,
        }

class TestResult:
    """Result of a single test execution"""
    def __init__(self, test_case: TestCase, query_result: QueryResult,
                 answer_correct: bool = False, cost: float = 0.0):
        self.test_case = test_case
        self.query_result = query_result
        self.answer_correct = answer_correct
        self.cost = cost

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_case": self.test_case.to_dict(),
            "query_result": self.query_result.to_dict(),
            "answer_correct": self.answer_correct,
            "cost": self.cost,
        }
    
class BenchmarkResult:
    """Complete results for a model run"""
    def __init__(self, configuration: Dict[str, Any],
                 test_results: Optional[List[TestResult]] = None,
                 timestamp: Optional[datetime] = None, total_cost: float = 0.0,
                 total_benchmark_time: float = 0.0, 
                 llm_formulate_kql_errors_total: int = 0,
                 average_llm_formulate_kql_errors_per_test: float = 0.0):
        self.configuration = configuration
        self.test_results = test_results if test_results is not None else []
        self.timestamp = timestamp if timestamp is not None else datetime.now()
        self.total_cost = total_cost
        self.total_benchmark_time = total_benchmark_time
        self.llm_formulate_kql_errors_total = llm_formulate_kql_errors_total
        self.average_llm_formulate_kql_errors_per_test = average_llm_formulate_kql_errors_per_test
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage"""
        if not self.test_results:
            return 0.0
        successful = sum(1 for r in self.test_results if r.answer_correct)
        return successful / len(self.test_results) * 100
    
    @property
    def statistics(self) -> Dict[str, Any]:
        """Get statistics summary as dict"""
        # Calculate execution times
        execution_times = [r.query_result.execution_time for r in self.test_results 
                           if r.query_result and r.query_result.execution_time]
        
        # Calculate average attempts
        attempts = [r.query_result.attempts for r in self.test_results 
                   if r.query_result and r.query_result.attempts]
        avg_attempts = sum(attempts) / len(attempts) if attempts else 0
        max_attempts = self.configuration.get("configured_max_tries", 0)
        
        total_time = sum(execution_times) if execution_times else 0
        avg_time = total_time / len(execution_times) if execution_times else 0

        # Calculate cost per test
        test_costs = [r.cost for r in self.test_results if r.cost is not None]
        avg_cost_per_test = sum(test_costs) / len(test_costs) if test_costs else 0
        
        return {
            "total_tests": len(self.test_results),
            "successful_tests": sum(1 for r in self.test_results if r.answer_correct),
            "success_rate": self.success_rate,
            "total_cost": self.total_cost,
            "average_cost_per_test": avg_cost_per_test,
            "total_execution_time": total_time,
            "avg_execution_time": avg_time,
            "total_benchmark_time": self.total_benchmark_time,
            "average_attempts": avg_attempts,
            "max_attempts": max_attempts,
            "llm_formulate_kql_errors_total": self.llm_formulate_kql_errors_total,
            "average_llm_formulate_kql_errors_per_test": self.average_llm_formulate_kql_errors_per_test,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "configuration": self.configuration,
            "test_results": [tr.to_dict() for tr in self.test_results],
            "timestamp": self.timestamp.isoformat(),
            "total_cost": self.total_cost,
            "total_benchmark_time": self.total_benchmark_time,
            "statistics": self.statistics,
        }
