# MTUCI Shop Detector

Person detection system using YOLO11 with analytics and PDF reporting.

## Quick Start

### Local Development

```bash
# Install dependencies
uv sync

# Start database
docker-compose -f docker-compose.db-only.yml up -d

# Run application
uv run streamlit run main.py
```

Open http://localhost:8501

### Docker (Full Stack)

```bash
# Create environment file
cp .env.example .env

# Start all services
docker-compose up -d
```

Open http://localhost:8501

## Features

- Video file processing with person detection
- Image batch processing
- Real-time statistics (min/max/avg person count)
- PostgreSQL analytics storage
- PDF report generation
- Object tracking support

## Tech Stack

- **ML**: YOLO11 (Ultralytics)
- **UI**: Streamlit
- **Database**: PostgreSQL
- **Reports**: ReportLab

## Database

Connect to PostgreSQL:
```bash
docker exec -it db_postgres psql -U postgres -d mtuci_shop_detector
```

View analytics:
```sql
SELECT * FROM detection_analytics ORDER BY id DESC LIMIT 10;
```

## Configuration

Edit `.env` file:
```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=mtuci_shop_detector
DB_USER=postgres
DB_PASSWORD=postgres
```

## Project Structure

```
├── main.py              # Streamlit application
├── database.py          # PostgreSQL manager
├── report_generator.py  # PDF reports
├── docker-compose.yml   # Full stack
└── docker-compose.db-only.yml  # Database only
```

## Stop Services

```bash
# Stop all
docker-compose down

# Stop database only
docker-compose -f docker-compose.db-only.yml down