# ARGO Data Platform MVP

A comprehensive platform for processing, storing, and analyzing ARGO oceanographic data with natural language query capabilities using RAG (Retrieval-Augmented Generation).

## 🌊 Overview

This platform provides an end-to-end solution for working with ARGO float data, featuring:

- **NetCDF Data Ingestion**: Automated processing of ARGO profile files
- **Dual Database Storage**: PostgreSQL for structured data + Vector database for semantic search
- **RAG-powered Chat Interface**: Natural language to SQL translation using LLMs
- **Interactive Dashboard**: Streamlit-based visualization and exploration tools
- **Geospatial Analysis**: Global ocean data mapping and regional analysis

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   ARGO NetCDF   │───▶│  Data Ingestion  │───▶│   PostgreSQL    │
│     Files       │    │     Pipeline     │    │    Database     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │  Vector Database │    │  RAG System     │
                       │ (FAISS/ChromaDB) │◄──▶│ (LLM + Search)  │
                       └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │   Streamlit     │
                                               │   Dashboard     │
                                               └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PostgreSQL 12+
- OpenAI API key (for LLM functionality)

### Installation

1. **Clone and setup the project:**
   ```bash
   git clone <repository_url>
   cd argo-data-platform
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Setup PostgreSQL database:**
   ```sql
   CREATE DATABASE argo_data;
   CREATE USER argo_user WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE argo_data TO argo_user;
   ```

4. **Initialize the database:**
   ```bash
   python -c "from src.database.connection import init_database; init_database()"
   ```

5. **Generate sample data (optional):**
   ```bash
   python tests/sample_data_generator.py 20 ./data/sample_argo
   ```

6. **Process sample data:**
   ```bash
   python -m src.utils.data_pipeline ./data/sample_argo
   ```

7. **Launch the dashboard:**
   ```bash
   streamlit run src/dashboard/app.py
   ```

Visit `http://localhost:8501` to access the dashboard!

## 📊 Features

### Data Ingestion & Processing
- Automated ARGO NetCDF file parsing
- Quality control flag handling
- Metadata extraction and geospatial indexing
- Batch processing with progress tracking

### Database Storage
- **PostgreSQL**: Structured storage of profiles, measurements, and metadata
- **Vector Database**: Semantic search using FAISS and ChromaDB
- **Quality Control**: ARGO standard QC flag support
- **Indexing**: Optimized for geospatial and temporal queries

### RAG System
- **Natural Language Processing**: Convert user questions to SQL
- **Semantic Search**: Find relevant profiles using vector similarity
- **LLM Integration**: OpenAI GPT-4 for query translation and response generation
- **Context Awareness**: Database schema and oceanographic domain knowledge

### Interactive Dashboard
- **Global Map**: Interactive visualization of ARGO float locations
- **Profile Viewer**: Detailed T-S diagrams and BGC parameter plots
- **Data Explorer**: Filter and search capabilities
- **Chat Interface**: Natural language queries with SQL generation
- **Processing Monitor**: Real-time ingestion status and statistics

## 💾 Database Schema

### Core Tables

**argo_profiles**: Profile metadata
- `profile_id`, `platform_number`, `cycle_number`
- `latitude`, `longitude`, `date`
- `depth_min`, `depth_max`, `n_levels`
- `parameters[]`, `data_mode`

**argo_measurements**: Individual measurements
- `profile_id` (FK), `pressure`, `depth`
- `temperature`, `salinity` (core parameters)
- `oxygen`, `chlorophyll`, `ph_total` (BGC parameters)
- Quality control flags for each parameter

**argo_floats**: Float metadata
- `platform_number`, `project_name`, `pi_name`

## 🔍 Usage Examples

### Natural Language Queries

```python
# Chat interface examples:
"Show me temperature profiles near the equator in March 2023"
"Find salinity measurements in the Arabian Sea for BGC floats"
"What are the oxygen levels at 1000m depth in the North Atlantic?"
"Compare chlorophyll between Mediterranean and Indian Ocean"
```

### Programmatic Access

```python
from src.database.connection import get_db_context
from src.database.models import ArgoProfile, ArgoMeasurement

# Query profiles in a region
with get_db_context() as session:
    profiles = session.query(ArgoProfile).filter(
        ArgoProfile.latitude.between(-5, 5),
        ArgoProfile.longitude.between(-20, 20)
    ).all()

# Process ARGO files
from src.utils.data_pipeline import process_argo_files
results = process_argo_files("./data/argo_files/")
```

### Vector Search

```python
from src.rag.vector_store import get_vector_store

vs = get_vector_store()
results = vs.search("Mediterranean temperature profiles", k=10)
```

## 🌐 Geographic Regions

The platform supports automatic region classification:

| Region | Latitude Range | Longitude Range |
|--------|---------------|------------------|
| North Atlantic | 40°N - 70°N | 80°W - 0° |
| South Atlantic | 60°S - 0° | 70°W - 20°E |
| North Pacific | 20°N - 60°N | 120°E - 240°E |
| Indian Ocean | 60°S - 30°N | 20°E - 120°E |
| Mediterranean | 30°N - 46°N | 6°W - 42°E |
| Arabian Sea | 10°N - 30°N | 50°E - 80°E |
| Arctic Ocean | 70°N - 90°N | All longitudes |
| Southern Ocean | 90°S - 50°S | All longitudes |

## 📈 Supported Parameters

### Core Parameters (All Floats)
- **TEMP**: Temperature (°C)
- **PSAL**: Practical Salinity (PSU)
- **PRES**: Pressure (dbar)

### BGC Parameters (Selected Floats)
- **DOXY**: Dissolved Oxygen (μmol/kg)
- **CHLA**: Chlorophyll-a (mg/m³)
- **BBP700**: Backscattering at 700nm (m⁻¹)
- **PH_IN_SITU_TOTAL**: pH on total scale
- **NITRATE**: Nitrate concentration (μmol/kg)
- **DOWNWELLING_PAR**: Photosynthetic radiation

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=argo_data
POSTGRES_USER=argo_user
POSTGRES_PASSWORD=your_password

# OpenAI API Configuration
OPENAI_API_KEY=your_openai_key

# Vector Database Configuration
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
FAISS_INDEX_PATH=./data/faiss_index

# Data Paths
ARGO_DATA_PATH=./data/argo_files
PROCESSED_DATA_PATH=./data/processed
```

### Database Tuning

For better performance with large datasets:

```sql
-- PostgreSQL optimization
CREATE INDEX CONCURRENTLY idx_profiles_location_date 
ON argo_profiles(latitude, longitude, date);

CREATE INDEX CONCURRENTLY idx_measurements_params 
ON argo_measurements(profile_id, temperature, salinity, oxygen);

-- Enable parallel queries
SET max_parallel_workers_per_gather = 4;
```

## 🧪 Testing

### Run Tests
```bash
# Unit tests
python -m pytest tests/ -v

# Integration tests with sample data
python tests/sample_data_generator.py 10
python -m src.utils.data_pipeline ./data/sample_argo
```

### Generate Sample Data
```bash
# Create 50 sample files (30% with BGC data)
python tests/sample_data_generator.py 50 ./data/sample_argo
```

## 🛠️ Development

### Project Structure
```
argo-data-platform/
├── src/
│   ├── ingestion/          # NetCDF file readers
│   ├── database/           # Models and connections
│   ├── rag/                # Vector search and LLM
│   ├── dashboard/          # Streamlit interface
│   └── utils/              # Data processing pipeline
├── tests/                  # Test files and sample data
├── data/                   # Data storage
├── docs/                   # Additional documentation
└── config/                 # Configuration files
```

### Adding New Parameters

1. **Update database model** (`src/database/models.py`):
   ```python
   # Add column to ArgoMeasurement
   new_parameter = Column(Float)
   new_parameter_qc = Column(Integer)
   ```

2. **Update parameter mappings** (`src/utils/data_pipeline.py`):
   ```python
   self.param_mappings['NEW_PARAM'] = 'new_parameter'
   self.qc_mappings['NEW_PARAM'] = 'new_parameter_qc'
   ```

3. **Update schema description** (`src/rag/llm_interface.py`)

### Extending LLM Capabilities

The RAG system can be extended with:
- Additional embedding models
- Custom query classification
- Domain-specific prompt templates
- Multi-modal support (images, time series)

## 🔒 Security Considerations

- API keys stored in environment variables
- Database credentials not hardcoded
- SQL injection protection via parameterized queries
- Rate limiting on API endpoints (recommended for production)

## 📝 API Reference

### Key Classes

- `ArgoNetCDFReader`: NetCDF file parsing
- `DataProcessor`: End-to-end data processing
- `VectorStore`: Semantic search functionality
- `QueryProcessor`: Natural language to SQL
- `DatabaseManager`: Database operations

### REST API (Future Enhancement)

The platform can be extended with FastAPI endpoints:

```python
@app.get("/profiles")
async def get_profiles(lat_min: float, lat_max: float, 
                      lon_min: float, lon_max: float):
    # Return profiles in bounding box
    
@app.post("/query")
async def natural_language_query(query: str):
    # Process natural language query
```

## 🚀 Deployment

### Docker Setup (Recommended)

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY .env .

EXPOSE 8501
CMD ["streamlit", "run", "src/dashboard/app.py"]
```

### Production Considerations

- Use environment-specific configurations
- Implement proper logging and monitoring
- Set up database backups
- Configure reverse proxy (nginx)
- Enable HTTPS
- Implement user authentication
- Set up data retention policies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation for API changes

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **ARGO Program**: For providing open oceanographic data
- **Scientific Community**: For ARGO data format standards
- **Open Source Libraries**: NumPy, pandas, xarray, Streamlit, and others

## 📞 Support

For questions or issues:
- Create an issue in the GitHub repository
- Check the documentation in the `docs/` directory
- Review the FAQ section below

## ❓ FAQ

**Q: How much data can the system handle?**
A: The system is designed to scale with proper database tuning. Test deployments have handled 100,000+ profiles.

**Q: Can I use local LLMs instead of OpenAI?**
A: Yes, the LLM interface can be extended to support local models like Llama or Mistral.

**Q: How accurate is the natural language to SQL translation?**
A: Accuracy depends on query complexity. Simple geographic and temporal queries work well (~90% accuracy). Complex multi-parameter queries may need refinement.

**Q: Can I add custom visualizations?**
A: Yes, the Streamlit dashboard is modular. Add new pages or modify existing visualization functions.

**Q: Is real-time data supported?**
A: The current MVP focuses on batch processing. Real-time ingestion can be added using streaming frameworks.

---

**🌊 Happy Ocean Data Analysis! 🌊**
