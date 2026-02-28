-- Create growth_predictions table
CREATE TABLE IF NOT EXISTS growth_predictions (
    id SERIAL PRIMARY KEY,
    plant_id VARCHAR(50) NOT NULL,
    user_id INTEGER,
    date_label VARCHAR(50),
    predicted_weight_g FLOAT,
    predicted_area_cm2 FLOAT,
    predicted_diameter_cm FLOAT,
    change_pct FLOAT,
    series_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_growth_prediction_plant ON growth_predictions(plant_id);
CREATE INDEX IF NOT EXISTS idx_growth_prediction_user ON growth_predictions(user_id);
CREATE INDEX IF NOT EXISTS idx_growth_prediction_created ON growth_predictions(created_at DESC);

-- Add soft delete columns to prediction_logs
ALTER TABLE prediction_logs ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP WITH TIME ZONE;
ALTER TABLE prediction_logs ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP;

-- Add soft delete columns to plant_scans
ALTER TABLE plant_scans ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP WITH TIME ZONE;
