"""
LLM Interface for RAG System

This module provides the interface to various LLM providers (OpenAI, local models)
for natural language to SQL translation and response generation.
"""

import os
import json
from typing import List, Dict, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime

# OpenAI integration
import openai
from openai import OpenAI

# Alternative: Local model support (uncomment if needed)
# from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from .vector_store import get_vector_store, Document
from ..database.connection import get_db_context, execute_raw_sql
from ..database.models import ArgoProfile, ArgoMeasurement

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QueryContext:
    """Context information for query processing"""
    user_query: str
    relevant_documents: List[Tuple[Document, float]]
    database_schema: Dict[str, Any]
    query_intent: str
    extracted_filters: Dict[str, Any]


class DatabaseSchemaProvider:
    """Provides database schema information for LLM context"""
    
    def __init__(self):
        self.schema_cache = None
    
    def get_schema_description(self) -> str:
        """Get a textual description of the database schema"""
        if self.schema_cache is None:
            self.schema_cache = self._build_schema_description()
        return self.schema_cache
    
    def _build_schema_description(self) -> str:
        """Build a comprehensive schema description"""
        schema_desc = '''
DATABASE SCHEMA FOR ARGO OCEANOGRAPHIC DATA:

1. ARGO_PROFILES table (main profile metadata):
   - profile_id (TEXT): Unique identifier for each profile (format: platform_number_cycle_number)
   - platform_number (TEXT): ARGO float identifier
   - cycle_number (INTEGER): Profile cycle number
   - latitude (FLOAT): Profile latitude in decimal degrees (-90 to 90)
   - longitude (FLOAT): Profile longitude in decimal degrees (-180 to 180)
   - date (TIMESTAMP): Profile collection date and time
   - depth_min, depth_max (FLOAT): Minimum and maximum depth of measurements in meters
   - n_levels (INTEGER): Number of measurement levels in the profile
   - data_mode (TEXT): 'R' (real-time), 'D' (delayed-mode), 'A' (adjusted)
   - parameters (TEXT[]): Array of available parameters (e.g., TEMP, PSAL, DOXY)

2. ARGO_MEASUREMENTS table (individual measurements):
   - profile_id (TEXT): Links to argo_profiles.profile_id
   - pressure (FLOAT): Pressure in decibars (dbar)
   - depth (FLOAT): Depth in meters
   - temperature (FLOAT): Temperature in degrees Celsius
   - salinity (FLOAT): Salinity in PSU (Practical Salinity Units)
   - oxygen (FLOAT): Dissolved oxygen in micromol/kg
   - chlorophyll (FLOAT): Chlorophyll concentration in mg/m³
   - bbp700 (FLOAT): Backscattering coefficient at 700nm in m⁻¹
   - ph_total (FLOAT): pH on total scale
   - nitrate (FLOAT): Nitrate concentration in micromol/kg
   - downwelling_par (FLOAT): Photosynthetically available radiation in microMoleQuanta/m²/sec
   - *_qc columns: Quality control flags (0-9) for each parameter

3. ARGO_FLOATS table (float metadata):
   - platform_number (TEXT): Float identifier
   - project_name (TEXT): Associated project
   - pi_name (TEXT): Principal investigator

GEOGRAPHIC REGIONS (for reference):
- North Atlantic: lat 40-70, lon -80-0
- South Atlantic: lat -60-0, lon -70-20
- North Pacific: lat 20-60, lon 120-240
- South Pacific: lat -60-0, lon 120-280
- Indian Ocean: lat -60-30, lon 20-120
- Arctic Ocean: lat 70-90
- Antarctic/Southern Ocean: lat -90--50
- Mediterranean Sea: lat 30-46, lon -6-42
- Arabian Sea: lat 10-30, lon 50-80
- Bay of Bengal: lat 5-25, lon 80-100

PARAMETER MEANINGS:
- TEMP: Temperature (°C)
- PSAL: Practical Salinity (PSU)
- PRES: Pressure (dbar)
- DOXY: Dissolved Oxygen (micromol/kg)
- CHLA: Chlorophyll-a (mg/m³)
- BBP700: Particulate backscattering coefficient at 700nm
- PH_IN_SITU_TOTAL: pH on total scale
- NITRATE: Nitrate concentration (micromol/kg)

QUERY GUIDELINES:
- Use ST_DWithin for distance-based queries with geography type
- Date filters should use proper timestamp comparison
- Quality flags: 1=good, 2=probably good, 3-4=questionable/bad, 9=missing
- Depth increases with pressure (surface=0, deep=2000+ meters)
'''
        return schema_desc


class LLMInterface:
    """Interface for Language Model operations"""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4"):
        self.provider = provider
        self.model = model
        self.client = None
        self.schema_provider = DatabaseSchemaProvider()
        
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required")
            self.client = OpenAI(api_key=api_key)
        
        logger.info(f"Initialized LLM interface: {provider} - {model}")
    
    def generate_sql_query(self, context: QueryContext) -> Tuple[str, str]:
        """
        Generate SQL query from natural language input
        
        Returns:
            Tuple of (sql_query, explanation)
        """
        
        # Build context for the LLM
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(context)
        
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,  # Low temperature for consistent SQL generation
                    max_tokens=1000
                )
                
                content = response.choices[0].message.content
                return self._parse_sql_response(content)
            
        except Exception as e:
            logger.error(f"Error generating SQL query: {e}")
            return "SELECT 1 as error", f"Error generating query: {str(e)}"
    
    def generate_response(self, query_results: List[Dict], 
                         original_query: str, sql_query: str) -> str:
        """
        Generate natural language response from query results
        """
        
        prompt = f"""
You are an oceanographic data analyst. Based on the following query results, 
provide a clear, informative response to the user's question.

Original Question: {original_query}

SQL Query Used: {sql_query}

Results (showing first 10 rows if more than 10):
{json.dumps(query_results[:10], indent=2, default=str)}

Total Results: {len(query_results)} records

Please provide a natural language summary that:
1. Answers the user's question directly
2. Highlights key findings or patterns
3. Mentions any limitations or caveats
4. Uses appropriate oceanographic terminology
5. Includes specific numbers/statistics where relevant

Keep the response concise but informative, suitable for both experts and non-experts.
"""
        
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert oceanographic data analyst providing insights from ARGO float data."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=800
                )
                
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Found {len(query_results)} results for your query, but encountered an error generating the detailed response."
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for SQL generation"""
        schema_desc = self.schema_provider.get_schema_description()
        
        return f"""You are an expert SQL assistant specialized in oceanographic ARGO float data queries.

{schema_desc}

INSTRUCTIONS:
1. Generate PostgreSQL-compatible SQL queries only
2. Always include proper JOINs when querying multiple tables
3. Use appropriate WHERE conditions for filtering
4. Include LIMIT clauses for large result sets (default 100)
5. Handle geographic queries properly (use appropriate bounding boxes)
6. Consider data quality by filtering on quality control flags when relevant
7. Use proper date/time filtering with BETWEEN or >= operators
8. Format dates as 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'

RESPONSE FORMAT:
Return your response as JSON with these fields:
{{
    "sql": "your SQL query here",
    "explanation": "brief explanation of what the query does",
    "confidence": "high|medium|low"
}}

EXAMPLES:
- "salinity profiles near the equator" → Filter by latitude BETWEEN -5 AND 5
- "March 2023" → Filter by date BETWEEN '2023-03-01' AND '2023-03-31'
- "Arabian Sea" → Filter by lat BETWEEN 10 AND 30 AND lon BETWEEN 50 AND 80
- "good quality data only" → Add WHERE temperature_qc = 1 AND salinity_qc = 1"""

    def _build_user_prompt(self, context: QueryContext) -> str:
        """Build the user prompt with context"""
        
        relevant_context = ""
        if context.relevant_documents:
            relevant_context = "RELEVANT CONTEXT FROM SIMILAR PROFILES:\n"
            for doc, score in context.relevant_documents[:3]:  # Top 3 matches
                relevant_context += f"- {doc.content} (similarity: {score:.3f})\n"
            relevant_context += "\n"
        
        return f"""
{relevant_context}USER QUERY: {context.user_query}

Generate a PostgreSQL query to answer this question about ARGO oceanographic data.
Focus on providing accurate, efficient queries that return meaningful oceanographic insights.
"""

    def _parse_sql_response(self, content: str) -> Tuple[str, str]:
        """Parse the LLM response to extract SQL and explanation"""
        try:
            # Try to parse as JSON first
            if content.strip().startswith('{'):
                data = json.loads(content)
                return data.get('sql', ''), data.get('explanation', '')
            
            # Fallback: extract SQL from markdown code blocks
            lines = content.split('\n')
            sql_lines = []
            explanation_lines = []
            in_sql_block = False
            
            for line in lines:
                if line.strip().startswith('```sql') or line.strip().startswith('```'):
                    in_sql_block = not in_sql_block
                    continue
                
                if in_sql_block:
                    sql_lines.append(line)
                else:
                    explanation_lines.append(line)
            
            sql = '\n'.join(sql_lines).strip()
            explanation = '\n'.join(explanation_lines).strip()
            
            return sql, explanation
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return content, "Generated query"


class QueryProcessor:
    """Main class for processing natural language queries"""
    
    def __init__(self, llm_provider: str = "openai", llm_model: str = "gpt-4"):
        self.llm = LLMInterface(llm_provider, llm_model)
        self.vector_store = get_vector_store()
    
    def process_query(self, user_query: str, max_results: int = 100) -> Dict[str, Any]:
        """
        Process a natural language query end-to-end
        
        Returns:
            Dictionary containing query results, SQL, explanation, and metadata
        """
        
        # Step 1: Retrieve relevant context from vector store
        relevant_docs = self.vector_store.search(user_query, k=5)
        
        # Step 2: Build query context
        context = QueryContext(
            user_query=user_query,
            relevant_documents=relevant_docs,
            database_schema={},  # Could be populated with dynamic schema info
            query_intent="",     # Could be classified by another model
            extracted_filters={}  # Could extract specific filters
        )
        
        # Step 3: Generate SQL query
        sql_query, explanation = self.llm.generate_sql_query(context)
        
        if not sql_query or sql_query.strip() == "":
            return {
                "error": "Could not generate SQL query",
                "user_query": user_query,
                "sql_query": "",
                "explanation": explanation,
                "results": [],
                "metadata": {}
            }
        
        # Step 4: Execute query
        try:
            # Add limit if not present
            if "LIMIT" not in sql_query.upper():
                sql_query += f" LIMIT {max_results}"
            
            results = execute_raw_sql(sql_query)
            
            # Convert results to list of dictionaries
            if results:
                columns = results[0].keys()
                result_dicts = [dict(zip(columns, row)) for row in results]
            else:
                result_dicts = []
            
        except Exception as e:
            logger.error(f"Error executing SQL query: {e}")
            return {
                "error": f"SQL execution error: {str(e)}",
                "user_query": user_query,
                "sql_query": sql_query,
                "explanation": explanation,
                "results": [],
                "metadata": {}
            }
        
        # Step 5: Generate natural language response
        nl_response = self.llm.generate_response(result_dicts, user_query, sql_query)
        
        # Step 6: Prepare final response
        return {
            "user_query": user_query,
            "sql_query": sql_query,
            "explanation": explanation,
            "results": result_dicts,
            "natural_language_response": nl_response,
            "metadata": {
                "result_count": len(result_dicts),
                "relevant_context_count": len(relevant_docs),
                "execution_time": datetime.now().isoformat(),
                "llm_provider": self.llm.provider,
                "llm_model": self.llm.model
            }
        }
    
    def get_query_suggestions(self, partial_query: str = "") -> List[str]:
        """Get query suggestions based on available data"""
        
        base_suggestions = [
            "Show me temperature profiles near the equator in the last month",
            "Find salinity measurements in the Arabian Sea for 2023",
            "What are the oxygen levels in the North Atlantic?",
            "Compare chlorophyll concentrations between seasons",
            "Show BGC parameters from the Mediterranean Sea",
            "Find the deepest ARGO profiles in the Pacific Ocean",
            "What ARGO floats are active near coordinates (25, -80)?",
            "Show quality-controlled temperature data from float 1234567",
            "Find profiles with both temperature and oxygen measurements",
            "What's the average salinity at 1000m depth globally?"
        ]
        
        if not partial_query:
            return base_suggestions
        
        # Filter suggestions based on partial query
        filtered = [s for s in base_suggestions if partial_query.lower() in s.lower()]
        return filtered if filtered else base_suggestions


# Global query processor instance
query_processor = None


def get_query_processor() -> QueryProcessor:
    """Get the global query processor instance"""
    global query_processor
    if query_processor is None:
        query_processor = QueryProcessor()
    return query_processor


def process_user_query(query: str) -> Dict[str, Any]:
    """Process a user query using the global processor"""
    processor = get_query_processor()
    return processor.process_query(query)


if __name__ == "__main__":
    # Test the RAG system
    processor = QueryProcessor()
    
    test_queries = [
        "Show me temperature profiles from the North Atlantic",
        "What's the salinity in the Arabian Sea in March 2023?",
        "Find oxygen measurements near the equator"
    ]
    
    for query in test_queries:
        print(f"\n=== Testing Query: {query} ===")
        result = processor.process_query(query)
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Generated SQL: {result['sql_query']}")
            print(f"Results: {len(result['results'])} records")
            print(f"Response: {result['natural_language_response'][:200]}...")
