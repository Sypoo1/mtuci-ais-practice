CREATE TABLE IF NOT EXISTS detection_analytics (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255),
    file_name VARCHAR(500),
    file_type VARCHAR(50),
    person_count INTEGER,
    person_count_min INTEGER,
    person_count_max INTEGER,
    person_count_avg FLOAT,
    confidence_threshold FLOAT,
    iou_threshold FLOAT,
    model_name VARCHAR(100)
);