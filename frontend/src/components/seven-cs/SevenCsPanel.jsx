import React, { useState, useEffect, useCallback } from 'react';
import styles from './seven-cs.module.css';

const SevenCsPanel = ({ sessionDeviceId }) => {
    const [analysisData, setAnalysisData] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [selectedDimension, setSelectedDimension] = useState(null);
    const [error, setError] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);

    // 7C framework definition for display
    const SEVEN_CS = {
        climate: {
            name: 'Climate',
            description: 'Emotional safety, respect, and comfort in group interactions',
            color: '#FFB74D',  // Warm orange
            icon: '🎧'
        },
        communication: {
            name: 'Communication',
            description: 'Quality and effectiveness of information exchange',
            color: '#64B5F6',  // Light blue
            icon: '📫'
        },
        compatibility: {
            name: 'Compatibility',
            description: 'How well group members\' working styles complement each other',
            color: '#BA68C8',  // Purple
            icon: '🤝'
        },
        conflict: {
            name: 'Conflict',
            description: 'Approaches to handling disagreements and contentious situations',
            color: '#EF5350',  // Red
            icon: '⚡'
        },
        context: {
            name: 'Context',
            description: 'Environmental factors and situational awareness',
            color: '#66BB6A',  // Green
            icon: '🌍'
        },
        contribution: {
            name: 'Contribution',
            description: 'Individual participation and effort balance',
            color: '#FFEE58',  // Yellow
            icon: '✋'
        },
        constructive: {
            name: 'Constructive',
            description: 'Goal achievement and mutual benefit',
            color: '#26C6DA',  // Cyan
            icon: '🎯'
        }
    };

    // Fetch analysis results on component mount
    useEffect(() => {
        console.log('[7Cs] Component mounted with sessionDeviceId:', sessionDeviceId);
        if (sessionDeviceId) {
            fetchAnalysisResults();
        }
    }, [sessionDeviceId]);

    const fetchAnalysisResults = async () => {
        if (!sessionDeviceId) return;

        console.log('[7Cs] Fetching results for sessionDeviceId:', sessionDeviceId);
        setIsLoading(true);
        setError(null);

        try {
            const url = `/api/v1/seven-cs/results/${sessionDeviceId}`;
            console.log('[7Cs] Fetching from URL:', url);
            const response = await fetch(url);
            const data = await response.json();
            console.log('[7Cs] Response status:', response.status, 'Data:', data);

            if (response.ok) {
                if (data.status === 'not_analyzed') {
                    // No analysis exists yet
                    console.log('[7Cs] No analysis exists yet');
                    setAnalysisData(null);
                } else {
                    console.log('[7Cs] Analysis data received');
                    setAnalysisData(data);
                }
            } else {
                console.error('[7Cs] Error response:', data);
                setError(data.error || 'Failed to fetch analysis results');
            }
        } catch (err) {
            console.error('[7Cs] Error fetching results:', err);
            setError('Failed to connect to server');
        } finally {
            setIsLoading(false);
        }
    };

    const triggerAnalysis = async () => {
        if (!sessionDeviceId) return;

        setIsAnalyzing(true);
        setError(null);

        try {
            const response = await fetch(`/api/v1/seven-cs/analyze/${sessionDeviceId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            const data = await response.json();

            if (response.ok) {
                // Start polling for results
                pollForResults();
            } else {
                setError(data.error || 'Failed to start analysis');
                setIsAnalyzing(false);
            }
        } catch (err) {
            console.error('Error triggering analysis:', err);
            setError('Failed to connect to server');
            setIsAnalyzing(false);
        }
    };

    const pollForResults = async () => {
        const maxAttempts = 60;  // Poll for max 5 minutes (60 * 5 seconds)
        let attempts = 0;

        const poll = async () => {
            if (attempts >= maxAttempts) {
                setError('Analysis timed out. Please try again.');
                setIsAnalyzing(false);
                return;
            }

            try {
                const response = await fetch(`/api/v1/seven-cs/status/${sessionDeviceId}`);
                const data = await response.json();

                if (data.status === 'completed') {
                    // Fetch full results
                    await fetchAnalysisResults();
                    setIsAnalyzing(false);
                } else if (data.status === 'failed') {
                    setError('Analysis failed. Please try again.');
                    setIsAnalyzing(false);
                } else {
                    // Still processing, continue polling
                    attempts++;
                    setTimeout(poll, 5000);  // Poll every 5 seconds
                }
            } catch (err) {
                console.error('Error polling status:', err);
                attempts++;
                setTimeout(poll, 5000);
            }
        };

        poll();
    };

    const handleDimensionClick = useCallback((dimension) => {
        setSelectedDimension(dimension === selectedDimension ? null : dimension);
    }, [selectedDimension]);

    const getScoreColor = (score) => {
        if (score >= 75) return '#4CAF50';  // Green
        if (score >= 50) return '#FF9800';  // Orange
        return '#F44336';  // Red
    };

    const renderDimensionCard = (dimension, data) => {
        const config = SEVEN_CS[dimension];
        const score = data?.score || 0;
        const count = analysisData?.counts?.[dimension] || 0;
        const isSelected = selectedDimension === dimension;

        return (
            <div
                key={dimension}
                className={`${styles.dimensionCard} ${isSelected ? styles.selected : ''}`}
                onClick={() => handleDimensionClick(dimension)}
                style={{ borderLeftColor: config.color }}
            >
                <div className={styles.cardHeader}>
                    <div className={styles.cardTitle}>
                        <span className={styles.cardIcon}>
            {typeof config.icon === "string" && config.icon.length < 6 ? (
                config.icon
            ) : (
                <img
                    src={config.icon}
                    alt={config.name}
                    className={styles.iconImage}
                />
            )}
        </span>
                        <h3>{config.name}</h3>
                    </div>
                    <span className={styles.segmentCount}>
                        {count} segments
                    </span>
                </div>

                <div className={styles.scoreSection}>
                    <div
                        className={styles.scoreCircle}
                        style={{ borderColor: getScoreColor(score) }}
                    >
                        <span className={styles.scoreValue}>{score}</span>
                        <span className={styles.scoreLabel}>/100</span>
                    </div>
                    <div className={styles.scoreBar}>
                        <div
                            className={styles.scoreProgress}
                            style={{
                                width: `${score}%`,
                                backgroundColor: getScoreColor(score)
                            }}
                        />
                    </div>
                </div>

                {data?.explanation && (
                    <p className={styles.explanation}>
                        {data.explanation.substring(0, 150)}
                        {data.explanation.length > 150 && '...'}
                    </p>
                )}
            </div>
        );
    };

    const renderSelectedDimensionDetails = () => {
        if (!selectedDimension || !analysisData) return null;

        const segments = analysisData.segments?.filter(
            s => s.dimension === selectedDimension
        ) || [];
        const dimensionData = analysisData.summary?.[selectedDimension];
        const config = SEVEN_CS[selectedDimension];

        return (
            <div className={styles.detailSection}>
                <div className={styles.detailHeader}>
                    <h3>
                        <span className={styles.cardIcon}>
        {typeof config.icon === "string" && config.icon.length < 6 ? (
          config.icon
        ) : (
          <img
            src={config.icon}
            alt={config.name}
            className={styles.iconImage}
          />
        )}
      </span>
                        {config.name} - {segments.length} coded segments
                    </h3>
                    <button
                        className={styles.closeButton}
                        onClick={() => setSelectedDimension(null)}
                    >
                        ✕
                    </button>
                </div>

                {dimensionData?.explanation && (
                    <div className={styles.fullExplanation}>
                        <h4>Analysis</h4>
                        <p>{dimensionData.explanation}</p>
                    </div>
                )}

                {dimensionData?.evidence && dimensionData.evidence.length > 0 && (
                    <div className={styles.evidenceSection}>
                        <h4>Key Evidence</h4>
                        <ul className={styles.evidenceList}>
                            {dimensionData.evidence.map((item, idx) => (
                                <li key={idx}>{item}</li>
                            ))}
                        </ul>
                    </div>
                )}

                <div className={styles.segmentList}>
                    <h4>Coded Segments</h4>
                    {segments.map((segment, idx) => (
                        <div key={idx} className={styles.codedSegment}>
                            <div className={styles.segmentMeta}>
                                <span className={styles.timestamp}>
                                    {formatTimestamp(segment.start_time)}
                                </span>
                                {segment.speaker_tag && (
                                    <span className={styles.speaker}>
                                        {segment.speaker_tag}
                                    </span>
                                )}
                                {segment.confidence && (
                                    <span className={styles.confidence}>
                                        {Math.round(segment.confidence * 100)}% confident
                                    </span>
                                )}
                            </div>
                            <div className={styles.segmentText}>
                                "{segment.text_snippet}"
                            </div>
                            {segment.coding_reason && (
                                <div className={styles.codingReason}>
                                    <em>{segment.coding_reason}</em>
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            </div>
        );
    };

    const formatTimestamp = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    const formatDate = (isoString) => {
        if (!isoString) return '';
        const date = new Date(isoString);
        return date.toLocaleString();
    };

    // Render loading state
    if (isLoading) {
        return (
            <div className={styles.container}>
                <div className={styles.loading}>
                    <div className={styles.spinner}></div>
                    <p>Loading analysis...</p>
                </div>
            </div>
        );
    }

    // Render error state
    if (error) {
        return (
            <div className={styles.container}>
                <div className={styles.error}>
                    <p>⚠️ {error}</p>
                    <button onClick={fetchAnalysisResults} className={styles.retryButton}>
                        Retry
                    </button>
                </div>
            </div>
        );
    }

    // Render no analysis state
    if (!analysisData || analysisData.status === 'not_analyzed') {
        return (
            <div className={styles.container}>
                <div className={styles.noAnalysis}>
                    <h3>No 7C Analysis Available</h3>
                    <p>Click below to analyze this session using the 7C framework</p>
                    <button
                        onClick={triggerAnalysis}
                        disabled={isAnalyzing}
                        className={styles.analyzeButton}
                    >
                        {isAnalyzing ? 'Analyzing...' : 'Run 7C Analysis'}
                    </button>
                </div>
            </div>
        );
    }

    // Render main analysis view
    return (
        <div className={styles.container}>
            <div className={styles.header}>
                <div className={styles.headerInfo}>
                    <h2>7C Collaboration Analysis</h2>
                    {analysisData.metadata && (
                        <span className={styles.lastUpdated}>
                            Last analyzed: {formatDate(analysisData.metadata.updated_at)}
                        </span>
                    )}
                </div>
                <button
                    onClick={triggerAnalysis}
                    disabled={isAnalyzing}
                    className={styles.updateButton}
                >
                    {isAnalyzing ? 'Analyzing...' : 'Update Analysis'}
                </button>
            </div>

            <div className={styles.dimensionGrid}>
                {Object.keys(SEVEN_CS).map(dimension =>
                    renderDimensionCard(
                        dimension,
                        analysisData?.summary?.[dimension]
                    )
                )}
            </div>

            {selectedDimension && renderSelectedDimensionDetails()}
        </div>
    );
};

export default SevenCsPanel;