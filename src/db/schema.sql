-- RESINET-IDAM: MySQL Production Database Schema
-- Database: resinet_db

CREATE DATABASE IF NOT EXISTS resinet_db;
USE resinet_db;

-- 1. Security Audit Logs Table
CREATE TABLE IF NOT EXISTS resinet_audit_logs (
    log_id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    event_type VARCHAR(64) NOT NULL,
    user_action VARCHAR(255) NOT NULL,
    details TEXT,
    status VARCHAR(32) DEFAULT 'SUCCESS',
    ip_source VARCHAR(64) DEFAULT '127.0.0.1'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 2. Intrusion Detection Alerts Table
CREATE TABLE IF NOT EXISTS resinet_detection_alerts (
    alert_id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    flow_id VARCHAR(64),
    detection_engine ENUM('MULTICLASS_SUPERVISED', 'OPEN_SET_ANOMALY', 'HYBRID') NOT NULL,
    predicted_class VARCHAR(64) NOT NULL,
    confidence_score DOUBLE DEFAULT 0.0,
    anomaly_score DOUBLE DEFAULT 0.0,
    threat_severity ENUM('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'BENIGN') NOT NULL,
    source_ip VARCHAR(64) DEFAULT 'N/A',
    dest_ip VARCHAR(64) DEFAULT 'N/A',
    dest_port INT DEFAULT 0
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 3. Model Benchmark Snapshots Table
CREATE TABLE IF NOT EXISTS resinet_model_benchmarks (
    benchmark_id INT AUTO_INCREMENT PRIMARY KEY,
    recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    dataset_name VARCHAR(64) NOT NULL,
    model_name VARCHAR(64) NOT NULL,
    macro_f1 DOUBLE NOT NULL,
    weighted_f1 DOUBLE NOT NULL,
    accuracy DOUBLE NOT NULL,
    roc_auc DOUBLE DEFAULT 0.0,
    execution_time_seconds DOUBLE DEFAULT 0.0,
    sha256_provenance VARCHAR(64) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

