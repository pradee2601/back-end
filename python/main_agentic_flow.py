import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import logging
from dotenv import load_dotenv
import asyncio
import difflib
import requests
import chromadb
from chromadb.utils import embedding_functions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


# Pydantic Models for structured outputs
class BMCSegments(BaseModel):
    value_proposition: str = Field(description="Core value proposition of the business")
    customer_segments: List[str] = Field(description="Target customer segments")
    channels: List[str] = Field(description="Distribution and communication channels")
    revenue_streams: List[str] = Field(description="Ways the business makes money")
    cost_structures: List[str] = Field(description="Main costs and expenses")
    key_resources: List[str] = Field(description="Essential resources needed")
    key_activities: List[str] = Field(description="Critical activities to perform")
    key_partnerships: List[str] = Field(description="Important partnerships and suppliers")
    customer_relationships: List[str] = Field(description="Types of customer relationships")
    

class ValidationReport(BaseModel):
    overall_feasibility_score: float = Field(description="Overall feasibility score 0-100")
    segment_scores: Dict[str, float] = Field(description="Individual segment scores")
    suggestions: List[str] = Field(description="Improvement suggestions")
    strengths: List[str] = Field(description="Identified strengths")
    risks: List[str] = Field(description="Potential risks")

class CompanyExample(BaseModel):
    company_name: str
    segment: str
    example_description: str
    relevance_score: float

class BMCVersion(BaseModel):
    version_id: str
    timestamp: str
    bmc_data: BMCSegments
    validation_report: ValidationReport
    changes_made: List[str]

class WorkflowState(BaseModel):
    business_idea: str = ""
    extracted_segments: Optional[BMCSegments] = None
    validation_report: Optional[ValidationReport] = None
    bmc_draft: Optional[BMCSegments] = None
    company_examples: List[CompanyExample] = []
    version_history: List[BMCVersion] = []
    current_step: str = "idea_extraction"
    errors: List[str] = []
    processing_time: Dict[str, float] = {}
    extraction_metrics: Dict[str, Any] = {}
    validation_metrics: Dict[str, Any] = {}
    bmc_editable: Dict[str, Any] = {}
    examples_markdown: str = ""
    version_diff: str = ""
    bmc_suggestions: Dict[str, str] = {}
    example_retrieval_metrics: Dict[str, Any] = {}
    version_control_metrics: Dict[str, Any] = {}
    bmc_markdown: str = ""
    bmc_generation_metrics: Dict[str, Any] = {}
# --- Gemini (Google Generative AI) imports ---


class AutoBMCSystem:
    def __init__(self, gemini_api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3,
            google_api_key=gemini_api_key
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=gemini_api_key
        )
        self.setup_knowledge_base()
        self.setup_chromadb()
        self.setup_workflow()
    
    def setup_knowledge_base(self):
        """Initialize RAG knowledge base with startup examples"""
        startup_examples = [
            "Airbnb: Value proposition - Unique travel experiences in local homes. Customer segments - Travelers seeking authentic experiences, Budget-conscious travelers. Revenue streams - Commission from bookings, Host fees.",
            "Dropbox: Value proposition - Simple file synchronization across devices. Customer segments - Individual users, Small businesses, Enterprise clients. Channels - Referral program, Direct sales, Partnerships.",
            "Slack: Value proposition - Team communication and collaboration platform. Revenue streams - Freemium model, Enterprise subscriptions. Customer relationships - Self-service, Dedicated support.",
            "Uber: Value proposition - On-demand transportation. Customer segments - Urban commuters, Occasional riders. Key activities - Platform maintenance, Driver recruitment, Marketing.",
            "Netflix: Value proposition - On-demand entertainment streaming. Revenue streams - Subscription model. Cost structures - Content acquisition, Technology infrastructure, Marketing.",
            "Spotify: Value proposition - Music streaming with personalization. Customer segments - Music lovers, Podcast listeners. Revenue streams - Premium subscriptions, Advertising.",
            "Tesla: Value proposition - Sustainable electric vehicles. Key resources - Battery technology, Manufacturing facilities. Key partnerships - Battery suppliers, Charging networks.",
            "Amazon: Value proposition - Everything store with fast delivery. Customer relationships - Self-service platform, Prime membership. Channels - Online platform, Physical stores.",
            "Google: Value proposition - Organize world's information. Revenue streams - Advertising, Cloud services, Hardware sales. Key activities - Search algorithm, Data processing."
        ]
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50
        )
        
        chunks = []
        for example in startup_examples:
            chunks.extend(text_splitter.split_text(example))
        
        self.knowledge_base = FAISS.from_texts(
            chunks,
            self.embeddings
        )
    
    def setup_chromadb(self):
        """Initialize ChromaDB for dynamic example storage and retrieval"""
        self.chroma_client = chromadb.Client()
        self.chroma_collection = self.chroma_client.create_collection(
            name="startup_examples",
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )

    def fetch_and_store_examples_tavily(self, bmc_segments: Dict[str, Any]):
        """Fetch startup examples from Tavily API and store in ChromaDB"""
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        headers = {"Authorization": f"Bearer {tavily_api_key}"}
        for segment, value in bmc_segments.items():
            if isinstance(value, list):
                query = f"startup example for {segment}: {' '.join(value[:2])}"
            else:
                query = f"startup example for {segment}: {value}"
            url = f"https://api.tavily.com/search"
            params = {"query": query, "num_results": 5}
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    for i, item in enumerate(data.get("results", [])):
                        doc_id = f"{segment}_{i}_{item.get('url', '')[:20]}"
                        self.chroma_collection.add(
                            documents=[item.get("content", "")],
                            metadatas=[{
                                "segment": segment,
                                "source": item.get("url", ""),
                                "title": item.get("title", ""),
                                "description": item.get("content", "")
                            }],
                            ids=[doc_id]
                        )
            except Exception as e:
                logger.error(f"Tavily fetch failed for {segment}: {e}")

    def setup_workflow(self):
        """Setup the LangGraph workflow"""
        workflow = StateGraph(WorkflowState)
        
        # Add nodes (agents)
        workflow.add_node("idea_extractor", self.idea_extractor_agent)
        workflow.add_node("segment_validator", self.segment_validator_agent)
        workflow.add_node("bmc_generator", self.bmc_generator_agent)
        workflow.add_node("example_retriever", self.example_retriever_agent)
        workflow.add_node("version_control", self.version_control_agent)
        
        # Define edges (autonomous flow)
        workflow.set_entry_point("idea_extractor")
        workflow.add_edge("idea_extractor", "segment_validator")
        workflow.add_edge("segment_validator", "bmc_generator")
        workflow.add_edge("bmc_generator", "example_retriever")
        workflow.add_edge("example_retriever", "version_control")
        workflow.add_edge("version_control", END)
        
        # Add conditional edges for error handling
        workflow.add_conditional_edges(
            "idea_extractor",
            self.should_continue,
            {
                "continue": "segment_validator",
                "retry": "idea_extractor",
                "end": END
            }
        )
        
        self.workflow = workflow.compile(checkpointer=MemorySaver())
    
    def should_continue(self, state: WorkflowState) -> str:
        """Autonomous decision making for workflow continuation"""
        if len(state.errors) > 3:
            return "end"
        elif state.extracted_segments is None and "idea_extraction" in [error for error in state.errors]:
            return "retry"
        else:
            return "continue"
    
    async def idea_extractor_agent(self, state: WorkflowState) -> WorkflowState:
        """Agent 1: Extract BMC segments from business idea (improved)"""
        start_time = time.time()
        logger.info("🔍 Idea Extractor Agent starting...")

        def preprocess_input(input_data):
            # Accepts either string (free-form) or dict (structured form)
            if isinstance(input_data, dict):
                # Concatenate all fields into a single string
                text = " ".join(str(v) for v in input_data.values() if v)
            else:
                text = str(input_data)
            # Simple filter: remove sentences with personal anecdotes or vague statements
            irrelevant_phrases = [
                "I remember", "my story", "when I was", "in my opinion", "personally", "I think", "my experience", "as a child", "once upon a time"
            ]
            filtered = []
            for sent in text.split('.'):
                if not any(phrase in sent for phrase in irrelevant_phrases):
                    filtered.append(sent)
            return '.'.join(filtered)

        try:
            parser = PydanticOutputParser(pydantic_object=BMCSegments)

            # Preprocess input for flexibility and filtering
            clean_idea = preprocess_input(state.business_idea)

            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert business analyst specializing in Business Model Canvas extraction.\nYour task is to analyze a business idea and extract all relevant BMC segments.\n\nExtract the following segments:\n- Value proposition (core value offered)\n- Customer segments (target customers)\n- Channels (how to reach customers)\n- Revenue streams (how to make money)  \n- Cost structures (main costs)\n- Key resources (essential resources)\n- Key activities (critical activities)\n- Key partnerships (important partners)\n- Customer relationships (relationship types)\n\nBe specific and actionable. If information is missing, make reasonable assumptions based on the business type.\n\n{format_instructions}"""),
                ("human", "Business Idea: {business_idea}")
            ])

            formatted_prompt = prompt.format_messages(
                business_idea=clean_idea,
                format_instructions=parser.get_format_instructions()
            )

            response = await self.llm.ainvoke(formatted_prompt)
            extracted_segments = parser.parse(response.content)

            # Calculate extraction metrics
            total_segments = 9
            filled = 0
            for field in extracted_segments.__fields_set__:
                value = getattr(extracted_segments, field)
                if isinstance(value, list):
                    if value and any(v.strip() for v in value):
                        filled += 1
                elif isinstance(value, str):
                    if value.strip():
                        filled += 1
            extraction_pct = (filled / total_segments) * 100

            # Store metrics in state
            state.extracted_segments = extracted_segments
            state.current_step = "segment_validation"
            state.processing_time["idea_extraction"] = time.time() - start_time
            state.extraction_metrics = {
                "segments_filled": filled,
                "total_segments": total_segments,
                "extraction_percentage": extraction_pct,
                "processing_time": state.processing_time["idea_extraction"]
            }

            logger.info(f"✅ Extracted segments in {state.processing_time['idea_extraction']:.2f}s | {extraction_pct:.1f}% segments filled")

        except Exception as e:
            state.errors.append(f"idea_extraction: {str(e)}")
            logger.error(f"❌ Idea extraction failed: {e}")

        return state
    
    async def segment_validator_agent(self, state: WorkflowState) -> WorkflowState:
        """Agent 2: Validate extracted segments (improved)"""
        start_time = time.time()
        logger.info("🔍 Segment Validator Agent starting...")

        try:
            parser = PydanticOutputParser(pydantic_object=ValidationReport)

            # Retrieve startup patterns for each segment (RAG)
            segments_dict = state.extracted_segments.dict() if state.extracted_segments else {}
            rag_examples = {}
            for segment, value in segments_dict.items():
                if isinstance(value, list):
                    query = f"{segment}: {' '.join(value[:2])}" if value else segment
                else:
                    query = f"{segment}: {value}"
                # Retrieve top 1 relevant example for context
                docs = self.knowledge_base.similarity_search(query, k=1)
                rag_examples[segment] = docs[0].page_content if docs else ""

            # Build a context string for the prompt
            pattern_context = "\n".join([
                f"{seg}: {rag_examples[seg]}" for seg in rag_examples if rag_examples[seg]
            ])

            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a startup validation expert. Analyze BMC segments for feasibility and market viability.\n\nFor each segment, compare it to real-world startup patterns and examples.\n\nStartup Patterns for Reference:\n{pattern_context}\n\nEvaluate each segment on:\n- Market demand and size\n- Competitive landscape\n- Resource requirements\n- Revenue potential\n- Implementation difficulty\n\nProvide:\n- Overall feasibility score (0-100)\n- Individual segment scores (0-100)\n- Specific improvement suggestions for any segment scoring below 70\n- Identified strengths\n- Potential risks\n\nBe critical but constructive. Focus on actionable insights.\n\n{{format_instructions}}"""),
                ("human", "BMC Segments to validate: {segments}")
            ])

            formatted_prompt = prompt.format_messages(
                segments=segments_dict,
                format_instructions=parser.get_format_instructions()
            )

            response = await self.llm.ainvoke(formatted_prompt)
            validation_report = parser.parse(response.content)

            # Calculate validation metrics
            segment_scores = validation_report.segment_scores
            validated_segments = len(segment_scores)
            total_segments = 9
            percent_validated = (validated_segments / total_segments) * 100
            low_score_segments = [seg for seg, score in segment_scores.items() if score < 70]

            # Store metrics in state
            state.validation_report = validation_report
            state.current_step = "bmc_generation"
            state.processing_time["segment_validation"] = time.time() - start_time
            state.validation_metrics = {
                "segments_validated": validated_segments,
                "total_segments": total_segments,
                "percent_validated": percent_validated,
                "low_score_segments": low_score_segments,
                "processing_time": state.processing_time["segment_validation"]
            }

            logger.info(f"✅ Validation completed in {state.processing_time['segment_validation']:.2f}s | {percent_validated:.1f}% segments validated")
            if low_score_segments:
                logger.info(f"⚠️ Segments needing improvement: {', '.join(low_score_segments)}")

        except Exception as e:
            state.errors.append(f"segment_validation: {str(e)}")
            logger.error(f"❌ Segment validation failed: {e}")

        return state
    
    async def bmc_generator_agent(self, state: WorkflowState) -> WorkflowState:
        """Agent 3: Generate polished BMC draft (improved)"""
        start_time = time.time()
        logger.info("🔍 BMC Generator Agent starting...")

        try:
            parser = PydanticOutputParser(pydantic_object=BMCSegments)

            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a Business Model Canvas expert. Create a polished, complete BMC based on extracted segments and validation feedback.\n\nImprove the segments by:\n- Incorporating validation suggestions\n- Adding specific, actionable details\n- Ensuring consistency across all blocks\n- Making each segment measurable and implementable\n- Addressing identified risks\n\nFor each BMC block, provide a short suggestion for how the user might edit or improve it.\n\nCreate a professional, investor-ready BMC that clearly articulates the business model.\n\n{format_instructions}"""),
                ("human", """
                Original Segments: {segments}
                
                Validation Report: {validation_report}
                
                Create an improved, final BMC draft.
                """)
            ])

            formatted_prompt = prompt.format_messages(
                segments=state.extracted_segments.dict(),
                validation_report=state.validation_report.dict(),
                format_instructions=parser.get_format_instructions()
            )

            response = await self.llm.ainvoke(formatted_prompt)
            bmc_draft = parser.parse(response.content)

            # Mark each block as editable and provide suggestions
            editable_bmc = {}
            suggestions = {}
            for field in bmc_draft.__fields_set__:
                value = getattr(bmc_draft, field)
                editable_bmc[field] = {
                    "value": value,
                    "editable": True
                }
                # Simple suggestion: prompt user to review and tailor for their context
                suggestions[field] = f"Review and tailor the {field.replace('_', ' ')} for your specific business context."

            # Generate Markdown table for human-readable output
            def to_md_table(bmc):
                rows = ["| Segment | Value |", "|---|---|"]
                for k, v in bmc.items():
                    val = v["value"]
                    if isinstance(val, list):
                        val = ", ".join(val)
                    rows.append(f"| {k.replace('_', ' ').title()} | {val} |")
                return "\n".join(rows)
            bmc_markdown = to_md_table(editable_bmc)

            # Store in state
            state.bmc_draft = bmc_draft
            state.bmc_editable = editable_bmc
            state.bmc_suggestions = suggestions
            state.bmc_markdown = bmc_markdown
            state.current_step = "example_retrieval"
            state.processing_time["bmc_generation"] = time.time() - start_time
            state.bmc_generation_metrics = {
                "processing_time": state.processing_time["bmc_generation"]
            }

            logger.info(f"✅ BMC generated in {state.processing_time['bmc_generation']:.2f}s (Markdown table ready)")

        except Exception as e:
            state.errors.append(f"bmc_generation: {str(e)}")
            logger.error(f"❌ BMC generation failed: {e}")

        return state
    
    async def example_retriever_agent(self, state: WorkflowState) -> WorkflowState:
        """Agent 4: Retrieve relevant startup examples using RAG (Tavily+ChromaDB)"""
        start_time = time.time()
        logger.info("🔍 Example Retriever Agent starting...")

        try:
            # Fetch and store fresh examples from Tavily if ChromaDB is empty
            bmc_dict = state.bmc_draft.dict()
            if self.chroma_collection.count() == 0:
                self.fetch_and_store_examples_tavily(bmc_dict)

            examples = []
            # Retrieve top 3 relevant examples per segment from ChromaDB
            for segment_name, segment_value in bmc_dict.items():
                if isinstance(segment_value, list):
                    search_query = f"{segment_name}: {' '.join(segment_value[:2])}"
                else:
                    search_query = f"{segment_name}: {segment_value}"
                results = self.chroma_collection.query(
                    query_texts=[search_query],
                    n_results=5,
                    where={"segment": segment_name}
                )
                for i in range(min(3, len(results["documents"][0]))):
                    doc = results["documents"][0][i]
                    meta = results["metadatas"][0][i]
                    example = CompanyExample(
                        company_name=meta.get("title", "Unknown"),
                        segment=segment_name,
                        example_description=meta.get("description", doc),
                        relevance_score=1.0  # Assume max relevance for web-fetched
                    )
                    examples.append(example)

            # Prepare Markdown table for UI
            def to_md_table(examples):
                rows = ["| Company | Segment | Description | Source |", "|---|---|---|---|"]
                for ex in examples:
                    source = ""
                    for v in self.chroma_collection.get(ids=[f"{ex.segment}_0_"])["metadatas"]:
                        if v.get("segment") == ex.segment:
                            source = v.get("source", "")
                            break
                    rows.append(f"| {ex.company_name} | {ex.segment} | {ex.example_description[:60]}... | {source} |")
                return "\n".join(rows)
            examples_markdown = to_md_table(examples)

            # Store in state
            state.company_examples = examples
            state.examples_markdown = examples_markdown
            state.current_step = "version_control"
            state.processing_time["example_retrieval"] = time.time() - start_time
            state.example_retrieval_metrics = {
                "total_examples": len(examples),
                "processing_time": state.processing_time["example_retrieval"]
            }

            logger.info(f"✅ Retrieved {len(examples)} examples from ChromaDB in {state.processing_time['example_retrieval']:.2f}s (Markdown table ready)")

        except Exception as e:
            state.errors.append(f"example_retrieval: {str(e)}")
            logger.error(f"❌ Example retrieval failed: {e}")

        return state
    
    async def version_control_agent(self, state: WorkflowState) -> WorkflowState:
        """Agent 5: Manage version control and history (improved)"""
        start_time = time.time()
        logger.info("🔍 Version Control Agent starting...")

        try:
            # Create new version
            version_id = f"v{len(state.version_history) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            timestamp = datetime.now().isoformat()

            # Compute diff from previous version (if any)
            def bmc_to_str(bmc):
                if not bmc:
                    return ""
                d = bmc.dict() if hasattr(bmc, 'dict') else bmc
                return json.dumps(d, sort_keys=True, indent=2)

            changes_made = []
            if len(state.version_history) == 0:
                changes_made = ["Initial BMC creation", "First validation completed", "Examples retrieved"]
                diff_summary = "Initial version."
            else:
                prev_bmc = state.version_history[-1].bmc_data
                prev_str = bmc_to_str(prev_bmc)
                curr_str = bmc_to_str(state.bmc_draft)
                diff = list(difflib.unified_diff(prev_str.splitlines(), curr_str.splitlines(), lineterm=""))
                diff_summary = "\n".join(diff) if diff else "No changes."
                changes_made = ["BMC refinement", "Updated validation", "Refreshed examples", "Diff computed"]

            new_version = BMCVersion(
                version_id=version_id,
                timestamp=timestamp,
                bmc_data=state.bmc_draft,
                validation_report=state.validation_report,
                changes_made=changes_made
            )

            # Enforce 10-version limit
            state.version_history.append(new_version)
            if len(state.version_history) > 10:
                state.version_history = state.version_history[-10:]

            # Store diff summary and metrics
            state.version_diff = diff_summary
            state.current_step = "completed"
            state.processing_time["version_control"] = time.time() - start_time
            state.version_control_metrics = {
                "total_versions": len(state.version_history),
                "processing_time": state.processing_time["version_control"]
            }

            logger.info(f"✅ Version {version_id} saved in {state.processing_time['version_control']:.2f}s (Diff computed)")

        except Exception as e:
            state.errors.append(f"version_control: {str(e)}")
            logger.error(f"❌ Version control failed: {e}")

        return state

    def rollback_to_version(self, state: WorkflowState, version_id: str) -> WorkflowState:
        """Restore a previous BMC version by version_id (improved)"""
        for version in state.version_history:
            if version.version_id == version_id:
                state.bmc_draft = version.bmc_data
                state.validation_report = version.validation_report
                state.current_step = "rollback_completed"
                logger.info(f"🔄 Rolled back to version {version_id}")
                return state
        state.errors.append(f"rollback: Version {version_id} not found")
        logger.error(f"❌ Rollback failed: Version {version_id} not found")
        return state
    
    async def process_business_idea(self, business_idea: str) -> Dict[str, Any]:
        """Main entry point - processes business idea through autonomous workflow"""
        logger.info("🚀 Starting autonomous BMC generation workflow...")
        
        initial_state = WorkflowState(business_idea=business_idea)
        
        # Run the autonomous workflow
        config = {"configurable": {"thread_id": f"bmc_{datetime.now().timestamp()}"}}
        
        final_state = await self.workflow.ainvoke(initial_state, config=config)
        
        # Calculate total processing time
        total_time = sum(final_state['processing_time'].values())
        
        # Prepare results
        results = {
            "success": len(final_state['errors']) == 0,
            "bmc_draft": final_state['bmc_draft'].dict() if final_state['bmc_draft'] else None,
            "validation_report": final_state['validation_report'].dict() if final_state['validation_report'] else None,
            "company_examples": [ex.dict() for ex in final_state['company_examples']],
            "version_history": [v.dict() for v in final_state['version_history']],
            "processing_times": final_state['processing_time'],
            "total_processing_time": total_time,
            "errors": final_state['errors']
        }
        
        logger.info(f"🎉 Workflow completed in {total_time:.2f}s with {len(final_state['errors'])} errors")
        
        return results

# Usage Example
async def main():
    """Example usage of the autonomous BMC system"""
    
    # Initialize system
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    system = AutoBMCSystem(gemini_api_key=gemini_api_key)
    
    # Example business idea
    business_idea = """
    I want to create a mobile app that connects local farmers directly with consumers in urban areas. 
    The app would allow farmers to list their fresh produce, and city residents can order directly 
    from farms within a 50-mile radius. We'd handle delivery logistics and take a commission. 
    The goal is to provide fresher produce to consumers while giving farmers better profit margins 
    by cutting out middlemen.
    """
    
    # Process autonomously
    results = await system.process_business_idea(business_idea)
    
    # Display results
    if results["success"]:
        print("🎉 BMC Generation Successful!")
        print(f"⏱️  Total processing time: {results['total_processing_time']:.2f}s")
        print(f"📊 Feasibility Score: {results['validation_report']['overall_feasibility_score']}")
        print(f"📚 Found {len(results['company_examples'])} relevant examples")
        print(f"📝 Version {results['version_history'][-1]['version_id']} created")
    else:
        print("❌ BMC Generation failed with errors:")
        for error in results["errors"]:
            print(f"   - {error}")
    
    return results

# FastAPI Integration (Optional)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse

app = FastAPI(title="Autonomous BMC Generator API")

class BusinessIdeaRequest(BaseModel):
    business_idea: str

class RollbackRequest(BaseModel):
    version_id: str
    state: dict

@app.post("/generate-bmc")
async def generate_bmc_endpoint(request: BusinessIdeaRequest):
    """API endpoint for BMC generation"""
    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        system = AutoBMCSystem(gemini_api_key=gemini_api_key)
        results = await system.process_business_idea(request.business_idea)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rollback-version")
async def rollback_version_endpoint(request: RollbackRequest):
    """API endpoint for rolling back to a previous BMC version and creating a new version entry."""
    try:
        # Reconstruct WorkflowState from dict
        state_data = request.state
        state = WorkflowState(**{k: v for k, v in state_data.items() if k in WorkflowState.__fields__})
        # Rebuild version_history as BMCVersion objects
        if 'version_history' in state_data:
            state.version_history = [BMCVersion(**v) for v in state_data['version_history']]
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        system = AutoBMCSystem(gemini_api_key=gemini_api_key)
        # Rollback
        state = system.rollback_to_version(state, request.version_id)
        if state.current_step == "rollback_completed":
            # Create a new version entry for the rollback
            version_id_new = f"v{len(state.version_history) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            timestamp = datetime.now().isoformat()
            new_version = BMCVersion(
                version_id=version_id_new,
                timestamp=timestamp,
                bmc_data=state.bmc_draft,
                validation_report=state.validation_report,
                changes_made=[f"Rollback to {request.version_id}"]
            )
            state.version_history.append(new_version)
            if len(state.version_history) > 10:
                state.version_history = state.version_history[-10:]
            state.current_step = "rollback_version_created"
            response = {
                'success': True,
                'bmc_draft': state.bmc_draft.dict() if state.bmc_draft else None,
                'validation_report': state.validation_report.dict() if state.validation_report else None,
                'version_history': [v.dict() for v in state.version_history],
                'current_step': state.current_step,
                'errors': state.errors
            }
        else:
            response = {
                'success': False,
                'errors': state.errors
            }
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())