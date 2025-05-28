import unittest
from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor
import threading # For potential thread-related debugging if needed

# Add benchmark root to sys.path to allow importing QuerySystem and TestCase
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from helpers.system import QuerySystem
from models.benchmark_models import TestCase, ModelConfig, QueryResult

class TestQuerySystemConcurrency(unittest.TestCase):

    def mock_azure_openai_call(self, messages: list, tool_schema: dict = None, **kwargs):
        """
        Mocks the _call_azure_openai method.
        It will return different responses based on the question in the messages
        and the function name in the tool_schema.
        """
        prompt_content = ""
        for msg in messages:
            if msg['role'] == 'user':
                prompt_content = msg['content']
                break
        
        # Mock responses for generate_query (KQLQuery tool)
        if tool_schema and tool_schema['function']['name'] == 'KQLQuery':
            if "T1003.001_Prompt" in prompt_content:
                # Simulate LLM returning KQL query and explanation
                return {"query": "T1003.001_KQL", "explanation": "Explanation for T1003.001"}
            elif "T1003.005_Prompt" in prompt_content:
                return {"query": "T1003.005_KQL", "explanation": "Explanation for T1003.005"}
        
        # Mock responses for check_if_result_contains_answer (ContainsAnswer tool)
        elif tool_schema and tool_schema['function']['name'] == 'ContainsAnswer':
            # Simulate LLM determining if results contain the answer
            if "T1003.001_Prompt" in prompt_content: # Check based on the question context
                return {"contains": True, "answer": "Out-Minidump.ps1", "result_summary": "Summary for T1003.001"}
            elif "T1003.005_Prompt" in prompt_content:
                return {"contains": True, "answer": "cmdkey.exe /list", "result_summary": "Summary for T1003.005"}
        
        # Fallback for unexpected calls or non-tool calls (though not expected in this test flow)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].message.content = "Fallback content"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 10
        mock_response.usage.prompt_tokens_details = MagicMock()
        mock_response.usage.prompt_tokens_details.cached_tokens = 0
        return mock_response


    @patch('helpers.system.LogAnalyticsConnector') # Mock the LA Connector
    @patch.object(QuerySystem, '_call_azure_openai') # Mock the OpenAI call method
    def test_concurrent_solve_calls_isolated(self, mock_openai_call_method, MockLogAnalyticsConnector):
        """
        Tests that two concurrent calls to QuerySystem.solve() process tests in isolation
        and map answers to the correct test IDs.
        """
        # Configure the mock for _call_azure_openai
        mock_openai_call_method.side_effect = self.mock_azure_openai_call

        # Configure the mock for LogAnalyticsConnector
        mock_la_querier_instance = MockLogAnalyticsConnector.return_value
        # Mock get_all_table_fields to return some dummy data to allow QuerySystem initialization
        mock_la_querier_instance.get_all_table_fields.return_value = {
            'DeviceEvents_CL': ['field1', 'field2'], 'SecurityAlert_CL': ['alert_field']
        }
        # Mock run_custom_query to return a dummy result (list of lists/tuples)
        # The first sublist is headers, subsequent are data rows.
        mock_la_querier_instance.run_custom_query.return_value = [
            ['Timestamp', 'Result'], # Header
            ['2023-01-01T12:00:00Z', 'dummy_data'] # Data row
        ]

        # Define ModelConfig (can be minimal for this test)
        model_config = ModelConfig(
            model_name="test-gpt-4o",
            model="gpt-4o", # actual model ID for pricing
            api_version="test_api_version",
            endpoint="test_endpoint",
            deployment_name="test_deployment",
            api_key_env="TEST_KEY",
            key_vault_url_env="TEST_VAULT",
            temperature=0.0,
            max_tokens=1000,
            tries=1
        )
        
        # Initialize QuerySystem - this will use the mocked LogAnalyticsConnector during __init__
        # We also need to mock AzureOpenAI client if it's created in QuerySystem.__init__
        # and any KeyVault calls. For simplicity, let's patch relevant os.getenv for key vault
        # and AzureOpenAI client directly if it simplifies setup.
        
        # Patching environment variables for KeyVault and AzureOpenAI client instantiation
        with patch.dict(os.environ, {
            "VAULT_URL": "https://fakevault.vault.azure.net",
            "AZURE_OPENAI_API_KEY_ENV_NAME": "DUMMY_KEY_NAME", # if QuerySystem uses this
            "AZURE_OPENAI_BASE_URL_ENV_NAME": "DUMMY_URL_NAME", # if QuerySystem uses this
            "WORKSPACE_ID_SECRET_NAME": "DUMMY_WORKSPACE_ID_NAME" # if QuerySystem uses this
        }), \
        patch('helpers.system.DefaultAzureCredential', MagicMock()) as MockAzureCredential, \
        patch('helpers.system.SecretClient') as MockSecretClient, \
        patch('helpers.system.AzureOpenAI') as MockAzureOpenAIClient:

            mock_secret_client_instance = MockSecretClient.return_value
            # Define a side effect for get_secret
            def get_secret_side_effect(secret_name):
                if secret_name == "WORKSPACE-ID":
                    return MagicMock(value="dummy_workspace_id")
                elif secret_name == "AZURE-OPENAI-API-KEY":
                    return MagicMock(value="dummy_api_key")
                elif secret_name == "AZURE-OPENAI-BASE-URL":
                    return MagicMock(value="dummy_base_url")
                return MagicMock(value="default_secret")
            mock_secret_client_instance.get_secret.side_effect = get_secret_side_effect
            
            # Mock AzureOpenAI client instance (though its chat.completions.create will be replaced by mock_openai_call_method)
            mock_openai_client_instance = MockAzureOpenAIClient.return_value 
            # The critical part is that QuerySystem._call_azure_openai is already patched by @patch.object

            query_system = QuerySystem(initial_config=model_config)

        # Define two test cases
        test_case_1 = TestCase(
            technique_id="T1003",
            technique_name="OS Credential Dumping",
            test_id="T1003.001",
            test_name="Dump LSASS Memory (Out-Minidump.ps1)",
            prompt="T1003.001_Prompt: How to dump LSASS memory using Out-Minidump.ps1?",
            answer="Out-Minidump.ps1", # Expected answer
            question_id="q1", # Unique ID for the question itself
            data_sources=["Process monitoring", "Command-line parameters"],
            platforms=["Windows"],
            permissions_required=["Administrator"],
            detection_logic="Monitor for execution of Out-Minidump.ps1 or related PowerShell commands that access LSASS process memory.",
            query_type="KQL"
        )

        test_case_2 = TestCase(
            technique_id="T1003",
            technique_name="OS Credential Dumping",
            test_id="T1003.005",
            test_name="Cached Domain Credentials (cmdkey)",
            prompt="T1003.005_Prompt: How to list cached domain credentials using cmdkey?",
            answer="cmdkey.exe /list", # Expected answer
            question_id="q2",
            data_sources=["Process monitoring", "Command-line parameters"],
            platforms=["Windows"],
            permissions_required=["User"],
            detection_logic="Monitor for execution of cmdkey.exe with parameters like /list.",
            query_type="KQL"
        )

        test_cases = [test_case_1, test_case_2]
        results = {} # To store results keyed by test_id or question_id

        def run_solve(qs, tc):
            # Simulate what run_test_case in main.py does, but simplified
            query_result = qs.solve(tc)
            return tc.test_id, query_result

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(run_solve, query_system, tc) for tc in test_cases]
            for future in futures:
                test_id, query_result = future.result()
                results[test_id] = query_result
        
        # Assertions
        self.assertIn("T1003.001", results)
        self.assertIn("T1003.005", results)

        # Check that the answer for T1003.001 is correct
        self.assertIsNotNone(results["T1003.001"])
        self.assertEqual(results["T1003.001"].answer, "Out-Minidump.ps1")
        self.assertEqual(results["T1003.001"].query, "T1003.001_KQL")


        # Check that the answer for T1003.005 is correct
        self.assertIsNotNone(results["T1003.005"])
        self.assertEqual(results["T1003.005"].answer, "cmdkey.exe /list")
        self.assertEqual(results["T1003.005"].query, "T1003.005_KQL")
        
        # Verify _call_azure_openai was called multiple times (2 for generate_query, 2 for check_if_result_contains_answer)
        self.assertEqual(mock_openai_call_method.call_count, 4) 


if __name__ == '__main__':
    unittest.main() 