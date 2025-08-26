"""
Simulation Analytics and Reporting System

This module provides comprehensive metrics collection, statistical analysis, and reporting
capabilities for the peer review simulation. It tracks all system interactions and provides
detailed analytics across all enhancement systems.
"""

import logging
import json
import csv
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import statistics
from concurrent.futures import ThreadPoolExecutor
import asyncio

from src.core.exceptions import ValidationError, SimulationError
from src.core.logging_config import get_logger
from src.data.enhanced_models import (
    EnhancedResearcher, StructuredReview, EnhancedVenue,
    ResearcherLevel, VenueType, BiasEffect
)

logger = get_logger(__name__)


@dataclass
class MetricsSnapshot:
    """Snapshot of simulation metrics at a specific time."""
    timestamp: datetime
    simulation_id: str
    
    # Core metrics
    total_researchers: int
    total_papers: int
    total_reviews: int
    total_venues: int
    
    # Review quality metrics
    avg_review_quality: float
    avg_review_length: int
    avg_confidence_level: float
    review_completion_rate: float
    
    # Bias metrics
    bias_effects_detected: Dict[str, int]
    avg_bias_impact: Dict[str, float]
    
    # Network metrics
    collaboration_density: float
    citation_network_size: int
    community_count: int
    
    # Career metrics
    tenure_success_rate: float
    promotion_rates: Dict[str, float]
    job_market_competition: float
    
    # Venue metrics
    venue_acceptance_rates: Dict[str, float]
    venue_quality_scores: Dict[str, float]
    
    # Strategic behavior metrics
    venue_shopping_incidents: int
    review_trading_detected: int
    citation_cartel_members: int
    salami_slicing_cases: int
    
    # Performance metrics
    processing_time_ms: float
    memory_usage_mb: float
    error_count: int


@dataclass
class AnalyticsConfiguration:
    """Configuration for analytics collection and reporting."""
    # Collection settings
    metrics_collection_interval: int = 60  # seconds
    detailed_logging_enabled: bool = True
    real_time_analytics: bool = True
    
    # Storage settings
    data_retention_days: int = 90
    export_format: str = "json"  # json, csv, parquet
    compression_enabled: bool = True
    
    # Analysis settings
    statistical_confidence_level: float = 0.95
    trend_analysis_window_days: int = 7
    anomaly_detection_threshold: float = 2.0  # standard deviations
    
    # Reporting settings
    generate_daily_reports: bool = True
    generate_weekly_summaries: bool = True
    include_visualizations: bool = True
    
    # Performance settings
    max_concurrent_analyses: int = 4
    batch_processing_size: int = 1000
    memory_limit_mb: int = 2048


class MetricsCollector:
    """Collects comprehensive metrics from all simulation systems."""
    
    def __init__(self, config: AnalyticsConfiguration):
        self.config = config
        self.metrics_history: List[MetricsSnapshot] = []
        self.current_metrics: Dict[str, Any] = {}
        self.collection_active = False
        
    def start_collection(self):
        """Start continuous metrics collection."""
        self.collection_active = True
        logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection."""
        self.collection_active = False
        logger.info("Metrics collection stopped")
    
    def collect_snapshot(self, simulation_coordinator) -> MetricsSnapshot:
        """Collect a complete metrics snapshot from all systems."""
        try:
            timestamp = datetime.now()
            
            # Collect core metrics
            core_metrics = self._collect_core_metrics(simulation_coordinator)
            
            # Collect review quality metrics
            review_metrics = self._collect_review_metrics(simulation_coordinator)
            
            # Collect bias metrics
            bias_metrics = self._collect_bias_metrics(simulation_coordinator)
            
            # Collect network metrics
            network_metrics = self._collect_network_metrics(simulation_coordinator)
            
            # Collect career metrics
            career_metrics = self._collect_career_metrics(simulation_coordinator)
            
            # Collect venue metrics
            venue_metrics = self._collect_venue_metrics(simulation_coordinator)
            
            # Collect strategic behavior metrics
            strategic_metrics = self._collect_strategic_behavior_metrics(simulation_coordinator)
            
            # Collect performance metrics
            performance_metrics = self._collect_performance_metrics(simulation_coordinator)
            
            snapshot = MetricsSnapshot(
                timestamp=timestamp,
                simulation_id=simulation_coordinator.state.simulation_id,
                **core_metrics,
                **review_metrics,
                **bias_metrics,
                **network_metrics,
                **career_metrics,
                **venue_metrics,
                **strategic_metrics,
                **performance_metrics
            )
            
            self.metrics_history.append(snapshot)
            logger.debug(f"Metrics snapshot collected at {timestamp}")
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Failed to collect metrics snapshot: {e}")
            raise SimulationError(f"Metrics collection failed: {e}")
    
    def _collect_core_metrics(self, coordinator) -> Dict[str, Any]:
        """Collect core simulation metrics."""
        return {
            'total_researchers': coordinator.state.total_researchers,
            'total_papers': coordinator.state.total_papers,
            'total_reviews': coordinator.state.total_reviews,
            'total_venues': coordinator.state.active_venues
        }
    
    def _collect_review_metrics(self, coordinator) -> Dict[str, Any]:
        """Collect review quality and completion metrics."""
        if not coordinator.review_system:
            return {
                'avg_review_quality': 0.0,
                'avg_review_length': 0,
                'avg_confidence_level': 0.0,
                'review_completion_rate': 0.0
            }
        
        # Get recent reviews for analysis
        recent_reviews = coordinator.review_system.get_recent_reviews(days=1)
        
        if not recent_reviews:
            return {
                'avg_review_quality': 0.0,
                'avg_review_length': 0,
                'avg_confidence_level': 0.0,
                'review_completion_rate': 0.0
            }
        
        quality_scores = [r.quality_score for r in recent_reviews if r.quality_score]
        lengths = [r.review_length for r in recent_reviews if r.review_length]
        confidence_levels = [r.confidence_level for r in recent_reviews if r.confidence_level]
        completed_reviews = [r for r in recent_reviews if r.completeness_score >= 0.8]
        
        return {
            'avg_review_quality': statistics.mean(quality_scores) if quality_scores else 0.0,
            'avg_review_length': int(statistics.mean(lengths)) if lengths else 0,
            'avg_confidence_level': statistics.mean(confidence_levels) if confidence_levels else 0.0,
            'review_completion_rate': len(completed_reviews) / len(recent_reviews) if recent_reviews else 0.0
        }
    
    def _collect_bias_metrics(self, coordinator) -> Dict[str, Any]:
        """Collect cognitive bias metrics."""
        if not coordinator.bias_engine:
            return {
                'bias_effects_detected': {},
                'avg_bias_impact': {}
            }
        
        bias_stats = coordinator.bias_engine.get_bias_statistics()
        
        return {
            'bias_effects_detected': bias_stats.get('effects_count', {}),
            'avg_bias_impact': bias_stats.get('average_impact', {})
        }
    
    def _collect_network_metrics(self, coordinator) -> Dict[str, Any]:
        """Collect social network metrics."""
        metrics = {
            'collaboration_density': 0.0,
            'citation_network_size': 0,
            'community_count': 0
        }
        
        if coordinator.collaboration_network:
            metrics['collaboration_density'] = coordinator.collaboration_network.calculate_network_density()
        
        if coordinator.citation_network:
            metrics['citation_network_size'] = coordinator.citation_network.get_network_size()
        
        if coordinator.conference_community:
            metrics['community_count'] = coordinator.conference_community.get_community_count()
        
        return metrics
    
    def _collect_career_metrics(self, coordinator) -> Dict[str, Any]:
        """Collect career progression metrics."""
        metrics = {
            'tenure_success_rate': 0.0,
            'promotion_rates': {},
            'job_market_competition': 0.0
        }
        
        if coordinator.tenure_track_manager:
            metrics['tenure_success_rate'] = coordinator.tenure_track_manager.get_success_rate()
        
        if coordinator.promotion_criteria_evaluator:
            metrics['promotion_rates'] = coordinator.promotion_criteria_evaluator.get_promotion_rates()
        
        if coordinator.job_market_simulator:
            metrics['job_market_competition'] = coordinator.job_market_simulator.get_competition_level()
        
        return metrics
    
    def _collect_venue_metrics(self, coordinator) -> Dict[str, Any]:
        """Collect venue performance metrics."""
        metrics = {
            'venue_acceptance_rates': {},
            'venue_quality_scores': {}
        }
        
        if coordinator.venue_registry:
            venues = coordinator.venue_registry.get_all_venues()
            for venue in venues:
                metrics['venue_acceptance_rates'][venue.id] = venue.acceptance_rate
                if hasattr(venue, 'quality_score'):
                    metrics['venue_quality_scores'][venue.id] = venue.quality_score
        
        return metrics
    
    def _collect_strategic_behavior_metrics(self, coordinator) -> Dict[str, Any]:
        """Collect strategic behavior detection metrics."""
        metrics = {
            'venue_shopping_incidents': 0,
            'review_trading_detected': 0,
            'citation_cartel_members': 0,
            'salami_slicing_cases': 0
        }
        
        if coordinator.venue_shopping_tracker:
            metrics['venue_shopping_incidents'] = coordinator.venue_shopping_tracker.get_incident_count()
        
        if coordinator.review_trading_detector:
            metrics['review_trading_detected'] = coordinator.review_trading_detector.get_detection_count()
        
        if coordinator.citation_cartel_detector:
            metrics['citation_cartel_members'] = coordinator.citation_cartel_detector.get_member_count()
        
        if coordinator.salami_slicing_detector:
            metrics['salami_slicing_cases'] = coordinator.salami_slicing_detector.get_case_count()
        
        return metrics
    
    def _collect_performance_metrics(self, coordinator) -> Dict[str, Any]:
        """Collect system performance metrics."""
        import psutil
        import time
        
        start_time = time.time()
        
        # Simulate some processing to measure performance
        _ = coordinator.get_system_health()
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB
        
        return {
            'processing_time_ms': processing_time,
            'memory_usage_mb': memory_usage,
            'error_count': coordinator.state.total_errors
        }


class StatisticalAnalyzer:
    """Performs statistical analysis on collected metrics."""
    
    def __init__(self, config: AnalyticsConfiguration):
        self.config = config
        
    def analyze_trends(self, metrics_history: List[MetricsSnapshot]) -> Dict[str, Any]:
        """Analyze trends in simulation metrics over time."""
        if len(metrics_history) < 2:
            return {'error': 'Insufficient data for trend analysis'}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([asdict(snapshot) for snapshot in metrics_history])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        trends = {}
        
        # Analyze numeric columns for trends
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column in ['timestamp']:
                continue
                
            values = df[column].dropna()
            if len(values) < 2:
                continue
            
            # Calculate trend metrics
            trend_data = self._calculate_trend_metrics(values)
            trends[column] = trend_data
        
        return trends
    
    def _calculate_trend_metrics(self, values: pd.Series) -> Dict[str, float]:
        """Calculate trend metrics for a series of values."""
        if len(values) < 2:
            return {}
        
        # Linear regression for trend
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        # Statistical measures
        mean_val = values.mean()
        std_val = values.std()
        min_val = values.min()
        max_val = values.max()
        
        # Trend direction
        trend_direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
        
        # Volatility (coefficient of variation)
        volatility = std_val / mean_val if mean_val != 0 else 0
        
        return {
            'slope': slope,
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'trend_direction': trend_direction,
            'volatility': volatility
        }
    
    def detect_anomalies(self, metrics_history: List[MetricsSnapshot]) -> Dict[str, List[Dict]]:
        """Detect anomalies in simulation metrics."""
        if len(metrics_history) < 10:  # Need sufficient data for anomaly detection
            return {}
        
        df = pd.DataFrame([asdict(snapshot) for snapshot in metrics_history])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        anomalies = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column in ['timestamp']:
                continue
            
            values = df[column].dropna()
            if len(values) < 10:
                continue
            
            # Z-score based anomaly detection
            z_scores = np.abs((values - values.mean()) / values.std())
            anomaly_indices = np.where(z_scores > self.config.anomaly_detection_threshold)[0]
            
            if len(anomaly_indices) > 0:
                column_anomalies = []
                for idx in anomaly_indices:
                    column_anomalies.append({
                        'timestamp': df.iloc[idx]['timestamp'].isoformat(),
                        'value': values.iloc[idx],
                        'z_score': z_scores.iloc[idx],
                        'expected_range': [
                            values.mean() - 2 * values.std(),
                            values.mean() + 2 * values.std()
                        ]
                    })
                anomalies[column] = column_anomalies
        
        return anomalies
    
    def calculate_correlations(self, metrics_history: List[MetricsSnapshot]) -> Dict[str, Dict[str, float]]:
        """Calculate correlations between different metrics."""
        if len(metrics_history) < 10:
            return {}
        
        df = pd.DataFrame([asdict(snapshot) for snapshot in metrics_history])
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        correlation_matrix = numeric_df.corr()
        
        # Convert to nested dictionary format
        correlations = {}
        for col1 in correlation_matrix.columns:
            correlations[col1] = {}
            for col2 in correlation_matrix.columns:
                if col1 != col2:  # Exclude self-correlation
                    correlations[col1][col2] = correlation_matrix.loc[col1, col2]
        
        return correlations
    
    def generate_statistical_summary(self, metrics_history: List[MetricsSnapshot]) -> Dict[str, Any]:
        """Generate comprehensive statistical summary."""
        if not metrics_history:
            return {}
        
        df = pd.DataFrame([asdict(snapshot) for snapshot in metrics_history])
        
        summary = {
            'data_points': len(metrics_history),
            'time_range': {
                'start': metrics_history[0].timestamp.isoformat(),
                'end': metrics_history[-1].timestamp.isoformat(),
                'duration_hours': (metrics_history[-1].timestamp - metrics_history[0].timestamp).total_seconds() / 3600
            },
            'descriptive_statistics': {},
            'trends': self.analyze_trends(metrics_history),
            'anomalies': self.detect_anomalies(metrics_history),
            'correlations': self.calculate_correlations(metrics_history)
        }
        
        # Add descriptive statistics for numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        for column in numeric_df.columns:
            if column not in ['timestamp']:
                values = numeric_df[column].dropna()
                if len(values) > 0:
                    summary['descriptive_statistics'][column] = {
                        'count': len(values),
                        'mean': values.mean(),
                        'median': values.median(),
                        'std': values.std(),
                        'min': values.min(),
                        'max': values.max(),
                        'q25': values.quantile(0.25),
                        'q75': values.quantile(0.75)
                    }
        
        return summary


class VisualizationGenerator:
    """Generates visualizations for simulation analytics."""
    
    def __init__(self, config: AnalyticsConfiguration):
        self.config = config
        
    def generate_trend_charts(self, metrics_history: List[MetricsSnapshot], 
                            output_dir: Path) -> List[str]:
        """Generate trend charts for key metrics."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if not metrics_history:
                return []
            
            # Set style
            plt.style.use('seaborn-v0_8')
            generated_files = []
            
            # Convert to DataFrame
            df = pd.DataFrame([asdict(snapshot) for snapshot in metrics_history])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Key metrics to visualize
            key_metrics = [
                'total_researchers', 'total_papers', 'total_reviews',
                'avg_review_quality', 'review_completion_rate',
                'collaboration_density', 'tenure_success_rate'
            ]
            
            for metric in key_metrics:
                if metric in df.columns and df[metric].notna().any():
                    plt.figure(figsize=(12, 6))
                    plt.plot(df['timestamp'], df[metric], marker='o', linewidth=2)
                    plt.title(f'{metric.replace("_", " ").title()} Over Time')
                    plt.xlabel('Time')
                    plt.ylabel(metric.replace("_", " ").title())
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    filename = f'trend_{metric}.png'
                    filepath = output_dir / filename
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    generated_files.append(str(filepath))
            
            return generated_files
            
        except ImportError:
            logger.warning("Matplotlib not available, skipping visualization generation")
            return []
        except Exception as e:
            logger.error(f"Failed to generate trend charts: {e}")
            return []
    
    def generate_correlation_heatmap(self, metrics_history: List[MetricsSnapshot],
                                   output_dir: Path) -> Optional[str]:
        """Generate correlation heatmap for metrics."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if len(metrics_history) < 10:
                return None
            
            df = pd.DataFrame([asdict(snapshot) for snapshot in metrics_history])
            numeric_df = df.select_dtypes(include=[np.number])
            
            # Calculate correlation matrix
            correlation_matrix = numeric_df.corr()
            
            # Generate heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f')
            plt.title('Metrics Correlation Heatmap')
            plt.tight_layout()
            
            filename = 'correlation_heatmap.png'
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available, skipping heatmap generation")
            return None
        except Exception as e:
            logger.error(f"Failed to generate correlation heatmap: {e}")
            return None
    
    def generate_distribution_plots(self, metrics_history: List[MetricsSnapshot],
                                  output_dir: Path) -> List[str]:
        """Generate distribution plots for key metrics."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if not metrics_history:
                return []
            
            generated_files = []
            df = pd.DataFrame([asdict(snapshot) for snapshot in metrics_history])
            
            # Key metrics for distribution analysis
            distribution_metrics = [
                'avg_review_quality', 'avg_confidence_level', 'processing_time_ms',
                'memory_usage_mb', 'collaboration_density'
            ]
            
            for metric in distribution_metrics:
                if metric in df.columns and df[metric].notna().any():
                    plt.figure(figsize=(10, 6))
                    
                    # Create subplot with histogram and box plot
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Histogram
                    ax1.hist(df[metric].dropna(), bins=20, alpha=0.7, edgecolor='black')
                    ax1.set_title(f'{metric.replace("_", " ").title()} Distribution')
                    ax1.set_xlabel(metric.replace("_", " ").title())
                    ax1.set_ylabel('Frequency')
                    
                    # Box plot
                    ax2.boxplot(df[metric].dropna())
                    ax2.set_title(f'{metric.replace("_", " ").title()} Box Plot')
                    ax2.set_ylabel(metric.replace("_", " ").title())
                    
                    plt.tight_layout()
                    
                    filename = f'distribution_{metric}.png'
                    filepath = output_dir / filename
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    generated_files.append(str(filepath))
            
            return generated_files
            
        except ImportError:
            logger.warning("Matplotlib not available, skipping distribution plots")
            return []
        except Exception as e:
            logger.error(f"Failed to generate distribution plots: {e}")
            return []


class ReportGenerator:
    """Generates comprehensive simulation reports."""
    
    def __init__(self, config: AnalyticsConfiguration):
        self.config = config
        self.analyzer = StatisticalAnalyzer(config)
        self.visualizer = VisualizationGenerator(config)
    
    def generate_comprehensive_report(self, metrics_history: List[MetricsSnapshot],
                                    output_dir: Path) -> Dict[str, str]:
        """Generate a comprehensive simulation report."""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate statistical analysis
            statistical_summary = self.analyzer.generate_statistical_summary(metrics_history)
            
            # Generate visualizations if enabled
            generated_charts = []
            if self.config.include_visualizations:
                generated_charts.extend(
                    self.visualizer.generate_trend_charts(metrics_history, output_dir)
                )
                generated_charts.extend(
                    self.visualizer.generate_distribution_plots(metrics_history, output_dir)
                )
                
                heatmap_file = self.visualizer.generate_correlation_heatmap(metrics_history, output_dir)
                if heatmap_file:
                    generated_charts.append(heatmap_file)
            
            # Generate report content
            report_content = self._generate_report_content(
                statistical_summary, generated_charts, metrics_history
            )
            
            # Save report
            report_files = {}
            
            # JSON report
            json_file = output_dir / 'simulation_report.json'
            with open(json_file, 'w') as f:
                json.dump({
                    'statistical_summary': statistical_summary,
                    'generated_charts': generated_charts,
                    'report_metadata': {
                        'generated_at': datetime.now().isoformat(),
                        'data_points': len(metrics_history),
                        'analysis_config': asdict(self.config)
                    }
                }, f, indent=2, default=str)
            report_files['json'] = str(json_file)
            
            # HTML report
            html_file = output_dir / 'simulation_report.html'
            with open(html_file, 'w') as f:
                f.write(self._generate_html_report(report_content, generated_charts))
            report_files['html'] = str(html_file)
            
            # CSV export of raw data
            if self.config.export_format in ['csv', 'all']:
                csv_file = output_dir / 'simulation_metrics.csv'
                df = pd.DataFrame([asdict(snapshot) for snapshot in metrics_history])
                df.to_csv(csv_file, index=False)
                report_files['csv'] = str(csv_file)
            
            logger.info(f"Comprehensive report generated in {output_dir}")
            return report_files
            
        except Exception as e:
            logger.error(f"Failed to generate comprehensive report: {e}")
            raise SimulationError(f"Report generation failed: {e}")
    
    def _generate_report_content(self, statistical_summary: Dict[str, Any],
                               generated_charts: List[str],
                               metrics_history: List[MetricsSnapshot]) -> Dict[str, Any]:
        """Generate structured report content."""
        if not metrics_history:
            return {'error': 'No metrics data available'}
        
        latest_snapshot = metrics_history[-1]
        
        return {
            'executive_summary': self._generate_executive_summary(statistical_summary, latest_snapshot),
            'key_findings': self._extract_key_findings(statistical_summary),
            'performance_analysis': self._analyze_performance(statistical_summary),
            'system_health': self._assess_system_health(statistical_summary, latest_snapshot),
            'recommendations': self._generate_recommendations(statistical_summary),
            'detailed_statistics': statistical_summary,
            'visualizations': generated_charts
        }
    
    def _generate_executive_summary(self, stats: Dict[str, Any], 
                                  latest: MetricsSnapshot) -> str:
        """Generate executive summary of simulation performance."""
        duration_hours = stats.get('time_range', {}).get('duration_hours', 0)
        data_points = stats.get('data_points', 0)
        
        summary = f"""
        Simulation Analytics Report
        ==========================
        
        Simulation ID: {latest.simulation_id}
        Analysis Period: {duration_hours:.1f} hours
        Data Points Collected: {data_points}
        
        Current Status:
        - Total Researchers: {latest.total_researchers}
        - Total Papers: {latest.total_papers}
        - Total Reviews: {latest.total_reviews}
        - Active Venues: {latest.total_venues}
        
        Key Performance Indicators:
        - Average Review Quality: {latest.avg_review_quality:.2f}/5.0
        - Review Completion Rate: {latest.review_completion_rate:.1%}
        - System Processing Time: {latest.processing_time_ms:.1f}ms
        - Memory Usage: {latest.memory_usage_mb:.1f}MB
        """
        
        return summary.strip()
    
    def _extract_key_findings(self, stats: Dict[str, Any]) -> List[str]:
        """Extract key findings from statistical analysis."""
        findings = []
        
        trends = stats.get('trends', {})
        anomalies = stats.get('anomalies', {})
        
        # Analyze trends
        for metric, trend_data in trends.items():
            if isinstance(trend_data, dict) and 'trend_direction' in trend_data:
                direction = trend_data['trend_direction']
                if direction != 'stable':
                    findings.append(f"{metric.replace('_', ' ').title()} is {direction}")
        
        # Report anomalies
        if anomalies:
            findings.append(f"Detected anomalies in {len(anomalies)} metrics")
        
        # Performance insights
        descriptive_stats = stats.get('descriptive_statistics', {})
        if 'processing_time_ms' in descriptive_stats:
            avg_time = descriptive_stats['processing_time_ms'].get('mean', 0)
            if avg_time > 1000:  # More than 1 second
                findings.append("System processing time is elevated")
        
        return findings
    
    def _analyze_performance(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze system performance metrics."""
        performance = {
            'overall_rating': 'Good',
            'bottlenecks': [],
            'efficiency_metrics': {}
        }
        
        descriptive_stats = stats.get('descriptive_statistics', {})
        
        # Analyze processing time
        if 'processing_time_ms' in descriptive_stats:
            time_stats = descriptive_stats['processing_time_ms']
            avg_time = time_stats.get('mean', 0)
            max_time = time_stats.get('max', 0)
            
            performance['efficiency_metrics']['avg_processing_time_ms'] = avg_time
            performance['efficiency_metrics']['max_processing_time_ms'] = max_time
            
            if avg_time > 1000:
                performance['bottlenecks'].append('High average processing time')
                performance['overall_rating'] = 'Needs Improvement'
        
        # Analyze memory usage
        if 'memory_usage_mb' in descriptive_stats:
            memory_stats = descriptive_stats['memory_usage_mb']
            avg_memory = memory_stats.get('mean', 0)
            max_memory = memory_stats.get('max', 0)
            
            performance['efficiency_metrics']['avg_memory_usage_mb'] = avg_memory
            performance['efficiency_metrics']['max_memory_usage_mb'] = max_memory
            
            if max_memory > 1024:  # More than 1GB
                performance['bottlenecks'].append('High memory usage detected')
        
        # Analyze error rates
        if 'error_count' in descriptive_stats:
            error_stats = descriptive_stats['error_count']
            avg_errors = error_stats.get('mean', 0)
            
            performance['efficiency_metrics']['avg_error_count'] = avg_errors
            
            if avg_errors > 5:
                performance['bottlenecks'].append('Elevated error rate')
                performance['overall_rating'] = 'Poor'
        
        return performance
    
    def _assess_system_health(self, stats: Dict[str, Any], 
                            latest: MetricsSnapshot) -> Dict[str, Any]:
        """Assess overall system health."""
        health = {
            'status': 'Healthy',
            'concerns': [],
            'metrics': {}
        }
        
        # Check review system health
        if latest.avg_review_quality < 2.0:
            health['concerns'].append('Low review quality detected')
            health['status'] = 'Warning'
        
        if latest.review_completion_rate < 0.8:
            health['concerns'].append('Low review completion rate')
            health['status'] = 'Warning'
        
        # Check performance health
        if latest.processing_time_ms > 2000:
            health['concerns'].append('High processing latency')
            health['status'] = 'Critical'
        
        if latest.memory_usage_mb > 2048:
            health['concerns'].append('High memory usage')
            health['status'] = 'Warning'
        
        # Check error rates
        if latest.error_count > 10:
            health['concerns'].append('High error count')
            health['status'] = 'Critical'
        
        health['metrics'] = {
            'review_quality_score': latest.avg_review_quality,
            'completion_rate': latest.review_completion_rate,
            'performance_score': min(100, max(0, 100 - (latest.processing_time_ms / 10))),
            'stability_score': min(100, max(0, 100 - latest.error_count))
        }
        
        return health
    
    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        trends = stats.get('trends', {})
        anomalies = stats.get('anomalies', {})
        descriptive_stats = stats.get('descriptive_statistics', {})
        
        # Performance recommendations
        if 'processing_time_ms' in descriptive_stats:
            avg_time = descriptive_stats['processing_time_ms'].get('mean', 0)
            if avg_time > 1000:
                recommendations.append("Consider optimizing system performance or increasing computational resources")
        
        # Quality recommendations
        if 'avg_review_quality' in descriptive_stats:
            avg_quality = descriptive_stats['avg_review_quality'].get('mean', 0)
            if avg_quality < 3.0:
                recommendations.append("Review quality is below average - consider reviewer training or bias adjustment")
        
        # Anomaly recommendations
        if anomalies:
            recommendations.append("Investigate detected anomalies to ensure system stability")
        
        # Trend recommendations
        declining_metrics = []
        for metric, trend_data in trends.items():
            if isinstance(trend_data, dict) and trend_data.get('trend_direction') == 'decreasing':
                if metric in ['avg_review_quality', 'review_completion_rate', 'tenure_success_rate']:
                    declining_metrics.append(metric)
        
        if declining_metrics:
            recommendations.append(f"Address declining trends in: {', '.join(declining_metrics)}")
        
        return recommendations
    
    def _generate_html_report(self, content: Dict[str, Any], 
                            charts: List[str]) -> str:
        """Generate HTML report with embedded visualizations."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Simulation Analytics Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .finding {{ background-color: #e8f4f8; padding: 10px; margin: 5px 0; border-radius: 3px; }}
                .recommendation {{ background-color: #fff3cd; padding: 10px; margin: 5px 0; border-radius: 3px; }}
                .chart {{ text-align: center; margin: 20px 0; }}
                .chart img {{ max-width: 100%; height: auto; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Simulation Analytics Report</h1>
                <p>Generated on: {timestamp}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <pre>{executive_summary}</pre>
            </div>
            
            <div class="section">
                <h2>Key Findings</h2>
                {key_findings_html}
            </div>
            
            <div class="section">
                <h2>System Health Assessment</h2>
                <p><strong>Status:</strong> {health_status}</p>
                {health_concerns_html}
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                {recommendations_html}
            </div>
            
            <div class="section">
                <h2>Performance Analysis</h2>
                <p><strong>Overall Rating:</strong> {performance_rating}</p>
                {performance_metrics_html}
            </div>
            
            {charts_html}
        </body>
        </html>
        """
        
        # Format findings
        findings_html = ""
        for finding in content.get('key_findings', []):
            findings_html += f'<div class="finding">{finding}</div>\n'
        
        # Format recommendations
        recommendations_html = ""
        for rec in content.get('recommendations', []):
            recommendations_html += f'<div class="recommendation">{rec}</div>\n'
        
        # Format health concerns
        health = content.get('system_health', {})
        concerns_html = ""
        for concern in health.get('concerns', []):
            concerns_html += f'<div class="finding">{concern}</div>\n'
        
        # Format performance metrics
        performance = content.get('performance_analysis', {})
        metrics_html = "<table><tr><th>Metric</th><th>Value</th></tr>"
        for metric, value in performance.get('efficiency_metrics', {}).items():
            metrics_html += f"<tr><td>{metric.replace('_', ' ').title()}</td><td>{value:.2f}</td></tr>"
        metrics_html += "</table>"
        
        # Format charts
        charts_html = ""
        if charts:
            charts_html = '<div class="section"><h2>Visualizations</h2>'
            for chart_path in charts:
                chart_name = Path(chart_path).stem.replace('_', ' ').title()
                charts_html += f'<div class="chart"><h3>{chart_name}</h3><img src="{Path(chart_path).name}" alt="{chart_name}"></div>'
            charts_html += '</div>'
        
        return html_template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            executive_summary=content.get('executive_summary', ''),
            key_findings_html=findings_html,
            health_status=health.get('status', 'Unknown'),
            health_concerns_html=concerns_html,
            recommendations_html=recommendations_html,
            performance_rating=performance.get('overall_rating', 'Unknown'),
            performance_metrics_html=metrics_html,
            charts_html=charts_html
        )


class SimulationAnalytics:
    """Main analytics system coordinating all analytics components."""
    
    def __init__(self, config: Optional[AnalyticsConfiguration] = None):
        self.config = config or AnalyticsConfiguration()
        self.metrics_collector = MetricsCollector(self.config)
        self.analyzer = StatisticalAnalyzer(self.config)
        self.report_generator = ReportGenerator(self.config)
        
        # Storage
        self.data_dir = Path("simulation_analytics")
        self.data_dir.mkdir(exist_ok=True)
        
        # Background collection
        self._collection_task = None
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_analyses)
        
        logger.info("Simulation analytics system initialized")
    
    def start_analytics(self, simulation_coordinator):
        """Start comprehensive analytics collection and processing."""
        self.simulation_coordinator = simulation_coordinator
        self.metrics_collector.start_collection()
        
        # Start background collection if real-time analytics enabled
        if self.config.real_time_analytics:
            self._start_background_collection()
        
        logger.info("Analytics system started")
    
    def stop_analytics(self):
        """Stop analytics collection and processing."""
        self.metrics_collector.stop_collection()
        
        if self._collection_task:
            self._collection_task.cancel()
        
        self._executor.shutdown(wait=True)
        logger.info("Analytics system stopped")
    
    def collect_metrics_snapshot(self) -> MetricsSnapshot:
        """Collect a single metrics snapshot."""
        return self.metrics_collector.collect_snapshot(self.simulation_coordinator)
    
    def generate_analytics_report(self, output_dir: Optional[Path] = None) -> Dict[str, str]:
        """Generate comprehensive analytics report."""
        if output_dir is None:
            output_dir = self.data_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        metrics_history = self.metrics_collector.metrics_history
        return self.report_generator.generate_comprehensive_report(metrics_history, output_dir)
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics."""
        if not self.metrics_collector.metrics_history:
            return {}
        
        latest_snapshot = self.metrics_collector.metrics_history[-1]
        return asdict(latest_snapshot)
    
    def export_metrics_data(self, format: str = "json", 
                          output_path: Optional[Path] = None) -> str:
        """Export collected metrics data."""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.data_dir / f"metrics_export_{timestamp}.{format}"
        
        metrics_data = [asdict(snapshot) for snapshot in self.metrics_collector.metrics_history]
        
        if format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
        elif format.lower() == "csv":
            df = pd.DataFrame(metrics_data)
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Metrics data exported to {output_path}")
        return str(output_path)
    
    def _start_background_collection(self):
        """Start background metrics collection task."""
        async def collection_loop():
            while self.metrics_collector.collection_active:
                try:
                    self.collect_metrics_snapshot()
                    await asyncio.sleep(self.config.metrics_collection_interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in background collection: {e}")
                    await asyncio.sleep(5)  # Brief pause before retry
        
        # Start the collection loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._collection_task = loop.create_task(collection_loop())
    
    def cleanup(self):
        """Cleanup analytics resources."""
        self.stop_analytics()
        
        # Clean up old data if retention policy is set
        if self.config.data_retention_days > 0:
            cutoff_date = datetime.now() - timedelta(days=self.config.data_retention_days)
            
            # Remove old snapshots
            self.metrics_collector.metrics_history = [
                snapshot for snapshot in self.metrics_collector.metrics_history
                if snapshot.timestamp > cutoff_date
            ]
            
            logger.info(f"Cleaned up analytics data older than {self.config.data_retention_days} days")