-- Initialize the RAG system database
-- This script will be run when the PostgreSQL container starts

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create workspaces table
CREATE TABLE IF NOT EXISTS workspaces (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    config JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create rides table (System workspace)
CREATE TABLE IF NOT EXISTS rides (
    id SERIAL PRIMARY KEY,
    ride_id UUID DEFAULT uuid_generate_v4(),
    ride_date DATE NOT NULL,
    driver_name VARCHAR(100) NOT NULL,
    passenger_count INTEGER CHECK (passenger_count > 0),
    distance_km DECIMAL(8,2) CHECK (distance_km > 0),
    fare_amount DECIMAL(10,2) CHECK (fare_amount > 0),
    pickup_location VARCHAR(255),
    dropoff_location VARCHAR(255),
    status VARCHAR(50) DEFAULT 'completed',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create platform engineering table (System workspace)
CREATE TABLE IF NOT EXISTS platform_eng (
    id SERIAL PRIMARY KEY,
    service_id UUID DEFAULT uuid_generate_v4(),
    service_name VARCHAR(100) NOT NULL,
    deployment_date DATE NOT NULL,
    status VARCHAR(50) DEFAULT 'active',
    cpu_usage DECIMAL(5,2) CHECK (cpu_usage >= 0 AND cpu_usage <= 100),
    memory_usage DECIMAL(5,2) CHECK (memory_usage >= 0 AND memory_usage <= 100),
    version VARCHAR(50),
    environment VARCHAR(50) DEFAULT 'production',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create metrics table (Both workspaces)
CREATE TABLE IF NOT EXISTS metrics (
    id SERIAL PRIMARY KEY,
    metric_id UUID DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(12,2) NOT NULL,
    metric_date DATE NOT NULL,
    category VARCHAR(50) NOT NULL,
    subcategory VARCHAR(50),
    unit VARCHAR(20),
    tags JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create COGS table (Custom workspace)
CREATE TABLE IF NOT EXISTS cogs (
    id SERIAL PRIMARY KEY,
    product_id UUID DEFAULT uuid_generate_v4(),
    product_name VARCHAR(100) NOT NULL,
    cost_per_unit DECIMAL(10,2) CHECK (cost_per_unit > 0),
    quantity INTEGER CHECK (quantity > 0),
    total_cost DECIMAL(12,2) CHECK (total_cost > 0),
    cost_date DATE NOT NULL,
    supplier VARCHAR(100),
    category VARCHAR(50),
    cost_type VARCHAR(50) DEFAULT 'direct',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create query logs table for tracking
CREATE TABLE IF NOT EXISTS query_logs (
    id SERIAL PRIMARY KEY,
    session_id UUID DEFAULT uuid_generate_v4(),
    user_query TEXT NOT NULL,
    workspace_type VARCHAR(50),
    sql_query TEXT,
    execution_time DECIMAL(8,3),
    result_count INTEGER,
    agent_path TEXT[],
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Insert default workspaces
INSERT INTO workspaces (name, description, config) VALUES
('system', 'System workspace for rides, platform engineering, and system metrics', 
 '{"tables": ["rides", "platform_eng", "metrics"], "color": "blue", "icon": "system"}'),
('custom', 'Custom workspace for COGS and business analysis', 
 '{"tables": ["cogs", "metrics"], "color": "green", "icon": "business"}')
ON CONFLICT (name) DO NOTHING;

-- Insert sample data

-- Sample rides data
INSERT INTO rides (ride_date, driver_name, passenger_count, distance_km, fare_amount, pickup_location, dropoff_location) VALUES
('2024-01-15', 'John Doe', 2, 15.5, 25.00, 'Downtown', 'Airport'),
('2024-01-16', 'Jane Smith', 1, 8.2, 12.50, 'Mall', 'University'),
('2024-01-17', 'Mike Johnson', 3, 22.1, 35.75, 'Hotel', 'Beach'),
('2024-01-18', 'Sarah Wilson', 1, 5.3, 8.00, 'Station', 'Office'),
('2024-01-19', 'Tom Brown', 2, 18.7, 28.50, 'Restaurant', 'Home'),
('2024-01-20', 'Lisa Davis', 4, 12.4, 22.00, 'Shopping Center', 'Park'),
('2024-01-21', 'Chris Lee', 1, 9.8, 14.75, 'Gym', 'Library'),
('2024-01-22', 'Anna Garcia', 2, 16.2, 24.50, 'Cinema', 'Cafe')
ON CONFLICT DO NOTHING;

-- Sample platform engineering data
INSERT INTO platform_eng (service_name, deployment_date, status, cpu_usage, memory_usage, version, environment) VALUES
('auth-service', '2024-01-10', 'active', 45.5, 78.2, 'v2.1.0', 'production'),
('user-service', '2024-01-12', 'active', 32.1, 65.4, 'v1.8.3', 'production'),
('payment-service', '2024-01-14', 'maintenance', 12.3, 45.7, 'v3.0.1', 'production'),
('notification-service', '2024-01-16', 'active', 28.9, 52.3, 'v1.5.2', 'production'),
('analytics-service', '2024-01-18', 'active', 67.2, 84.1, 'v2.3.0', 'production'),
('search-service', '2024-01-20', 'degraded', 89.4, 91.7, 'v1.9.5', 'production'),
('cache-service', '2024-01-22', 'active', 23.6, 38.9, 'v4.1.2', 'production')
ON CONFLICT DO NOTHING;

-- Sample metrics data
INSERT INTO metrics (metric_name, metric_value, metric_date, category, subcategory, unit) VALUES
('daily_active_users', 15420, '2024-01-15', 'user_engagement', 'daily', 'count'),
('api_response_time', 245.6, '2024-01-15', 'performance', 'api', 'milliseconds'),
('error_rate', 0.12, '2024-01-15', 'reliability', 'errors', 'percentage'),
('total_revenue', 12580.50, '2024-01-15', 'business', 'revenue', 'dollars'),
('conversion_rate', 3.45, '2024-01-15', 'business', 'conversion', 'percentage'),
('daily_active_users', 16230, '2024-01-16', 'user_engagement', 'daily', 'count'),
('api_response_time', 238.2, '2024-01-16', 'performance', 'api', 'milliseconds'),
('error_rate', 0.08, '2024-01-16', 'reliability', 'errors', 'percentage'),
('total_revenue', 13420.75, '2024-01-16', 'business', 'revenue', 'dollars'),
('conversion_rate', 3.78, '2024-01-16', 'business', 'conversion', 'percentage')
ON CONFLICT DO NOTHING;

-- Sample COGS data
INSERT INTO cogs (product_name, cost_per_unit, quantity, total_cost, cost_date, supplier, category) VALUES
('Product A - Widget', 5.50, 100, 550.00, '2024-01-15', 'Supplier Corp', 'Electronics'),
('Product B - Gadget', 12.25, 75, 918.75, '2024-01-16', 'Tech Solutions', 'Electronics'),
('Product C - Component', 8.90, 200, 1780.00, '2024-01-17', 'Parts Plus', 'Hardware'),
('Product D - Assembly', 15.75, 50, 787.50, '2024-01-18', 'Manufacturer Inc', 'Electronics'),
('Product E - Material', 3.20, 500, 1600.00, '2024-01-19', 'Raw Materials Co', 'Materials'),
('Product F - Tool', 22.50, 25, 562.50, '2024-01-20', 'Tool Supply', 'Tools'),
('Product G - Service', 45.00, 10, 450.00, '2024-01-21', 'Service Provider', 'Services'),
('Product H - Software', 99.99, 12, 1199.88, '2024-01-22', 'Software House', 'Software')
ON CONFLICT DO NOTHING;

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_rides_date ON rides(ride_date);
CREATE INDEX IF NOT EXISTS idx_rides_driver ON rides(driver_name);
CREATE INDEX IF NOT EXISTS idx_platform_eng_service ON platform_eng(service_name);
CREATE INDEX IF NOT EXISTS idx_platform_eng_status ON platform_eng(status);
CREATE INDEX IF NOT EXISTS idx_metrics_name_date ON metrics(metric_name, metric_date);
CREATE INDEX IF NOT EXISTS idx_metrics_category ON metrics(category);
CREATE INDEX IF NOT EXISTS idx_cogs_product ON cogs(product_name);
CREATE INDEX IF NOT EXISTS idx_cogs_date ON cogs(cost_date);
CREATE INDEX IF NOT EXISTS idx_query_logs_created_at ON query_logs(created_at);

-- Create a view for combined business metrics
CREATE OR REPLACE VIEW business_summary AS
SELECT 
    DATE_TRUNC('day', r.ride_date) as date,
    COUNT(r.id) as total_rides,
    SUM(r.fare_amount) as total_ride_revenue,
    AVG(r.fare_amount) as avg_fare,
    SUM(c.total_cost) as total_cogs,
    (SUM(r.fare_amount) - SUM(c.total_cost)) as gross_profit
FROM rides r
FULL OUTER JOIN cogs c ON DATE(r.ride_date) = DATE(c.cost_date)
GROUP BY DATE_TRUNC('day', COALESCE(r.ride_date, c.cost_date))
ORDER BY date;

-- Create a function to log queries
CREATE OR REPLACE FUNCTION log_query(
    p_user_query TEXT,
    p_workspace_type VARCHAR(50) DEFAULT NULL,
    p_sql_query TEXT DEFAULT NULL,
    p_execution_time DECIMAL(8,3) DEFAULT NULL,
    p_result_count INTEGER DEFAULT NULL,
    p_agent_path TEXT[] DEFAULT NULL,
    p_success BOOLEAN DEFAULT true,
    p_error_message TEXT DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    log_id UUID;
BEGIN
    INSERT INTO query_logs (
        user_query, workspace_type, sql_query, execution_time,
        result_count, agent_path, success, error_message
    ) VALUES (
        p_user_query, p_workspace_type, p_sql_query, p_execution_time,
        p_result_count, p_agent_path, p_success, p_error_message
    ) RETURNING session_id INTO log_id;
    
    RETURN log_id;
END;
$$ LANGUAGE plpgsql;