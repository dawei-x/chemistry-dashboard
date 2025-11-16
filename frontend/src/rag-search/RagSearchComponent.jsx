import React, { useState } from 'react';
import './rag-search.css';
import ReactMarkdown from 'react-markdown';

function RagSearchComponent() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [searchType, setSearchType] = useState('all');
  const [manualInsights, setManualInsights] = useState(null);
  const [loadingManualInsights, setLoadingManualInsights] = useState(false);

  const quickActions = [
    { label: 'Analytical Moments', query: 'most analytical thinking' },
    { label: 'Collaboration Patterns', query: 'effective collaboration patterns' },
    { label: 'Why Question', query: 'why do some discussions have higher engagement?' }
  ];

  const handleSearch = async (searchQuery = query) => {
    if (!searchQuery.trim()) return;
    
    setLoading(true);
    setManualInsights(null); // Clear manual insights on new search
    
    try {
      const response = await fetch('http://localhost:5002/api/v1/rag/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: searchQuery,
          filter_to_user: searchType === 'my_sessions'
        })
      });
      
      const data = await response.json();
      console.log('Search response:', data);
      setResults(data);
    } catch (error) {
      console.error('Search failed:', error);
      setResults({ 
        query_type: 'error',
        error: 'Search failed. Please try again.' 
      });
    }
    setLoading(false);
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const generateManualInsights = async () => {
    setLoadingManualInsights(true);

    try {
      const response = await fetch('http://localhost:5002/api/v1/rag/insights', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: results.query,
          search_results: {
            query: results.query,
            results: results.results,
            total_found: results.total_found
          }
        })
      });
      
      const data = await response.json();
      setManualInsights(data.insights);
    } catch (error) {
      console.error('Insights error:', error);
      setManualInsights('Failed to generate insights. Please try again.');
    }
    
    setLoadingManualInsights(false);
  };

  // Determine which insights to show: auto-generated or manual
  const displayedInsights = results?.insights || manualInsights;
  const hasResults = results?.results && results.results.length > 0;
  const showInsightsButton = hasResults && !results?.insights && !displayedInsights;

  return (
    <div className="rag-search-container">
      <div className="search-header">
        <h2>Discover Insights</h2>
        <p>Ask questions about discussion patterns and insights</p>
      </div>

      <div className="search-box">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
          placeholder="Ask anything about your discussions..."
          className="search-input"
        />
        <button 
          onClick={() => handleSearch()} 
          disabled={loading}
          className="search-button"
        >
          {loading ? 'Searching...' : 'Search'}
        </button>
      </div>

      <div className="search-filters">
        <label>
          <input
            type="radio"
            value="all"
            checked={searchType === 'all'}
            onChange={(e) => setSearchType(e.target.value)}
          />
          All Sessions
        </label>
        <label>
          <input
            type="radio"
            value="my_sessions"
            checked={searchType === 'my_sessions'}
            onChange={(e) => setSearchType(e.target.value)}
          />
          My Sessions Only
        </label>
      </div>

      <div className="quick-actions">
        <span>Try: </span>
        {quickActions.map((action, idx) => (
          <button
            key={idx}
            onClick={() => {
              setQuery(action.query);
              handleSearch(action.query);
            }}
            className="quick-action-btn"
          >
            {action.label}
          </button>
        ))}
      </div>

      {results && (
        <div className="search-results">
          {/* Error State */}
          {results.query_type === 'error' && (
            <div className="error">
              {results.error}
              {results.example && <p className="error-example">Example: {results.example}</p>}
            </div>
          )}

          {/* Insights Display - auto-generated or manual */}
          {displayedInsights && (
            <div className="insights-section">
              <h3>
                {results.insights ? 'AI Insights' : 'Generated Insights'}
              </h3>
              <div className="insights-box">
                <ReactMarkdown>{displayedInsights}</ReactMarkdown>
              </div>
            </div>
          )}

          {/* Results with Manual Insights Button */}
          {hasResults && (
            <>
              <h3>
                {displayedInsights ? 'Supporting Evidence' : 'Results'} ({results.total_found})
              </h3>
              
              {/* Show button only if no insights exist yet */}
              {showInsightsButton && (
                <button 
                  onClick={generateManualInsights}
                  disabled={loadingManualInsights}
                  className="insight-button"
                >
                  {loadingManualInsights ? 'Generating...' : 'Generate AI Insights'}
                </button>
              )}

              {results.results.map((result, idx) => (
                <div key={idx} className="result-item">
                  <div className="result-header">
                    <span className="session-badge">
                      Session {result.metadata.session_device_id}
                    </span>
                    <span className="time-badge">
                      {formatTime(result.metadata.start_time)} - {formatTime(result.metadata.end_time)}
                    </span>
                    <span className="speakers-badge">
                      {result.metadata.speaker_count} speakers
                    </span>
                  </div>
                  
                  <div className="result-text">
                    {result.text.substring(0, 300)}...
                  </div>
                  
                  <div className="result-metrics">
                    <span className="metric-item">
                      <span className="metric-label">Tone:</span>
                      <span className="metric-value">{result.metadata.avg_emotional_tone?.toFixed(0) || 0}</span>
                    </span>
                    <span className="metric-item">
                      <span className="metric-label">Analytic:</span>
                      <span className="metric-value">{result.metadata.avg_analytic_thinking?.toFixed(0) || 0}</span>
                    </span>
                    <span className="metric-item">
                      <span className="metric-label">Clout:</span>
                      <span className="metric-value">{result.metadata.avg_clout?.toFixed(0) || 0}</span>
                    </span>
                    <span className="metric-item">
                      <span className="metric-label">Authenticity:</span>
                      <span className="metric-value">{result.metadata.avg_authenticity?.toFixed(0) || 0}</span>
                    </span>
                    <span className="metric-item">
                      <span className="metric-label">Certainty:</span>
                      <span className="metric-value">{result.metadata.avg_certainty?.toFixed(0) || 0}</span>
                    </span>
                  </div>
                  
                  <a 
                    href={`/sessions/${Math.floor(result.metadata.session_device_id/10)}/pods/${result.metadata.session_device_id}/transcripts?highlight_time=${result.metadata.start_time}`}
                    className="view-link"
                  >
                    View in context →
                  </a>
                </div>
              ))}
            </>
          )}

          {/* No Results */}
          {results.query_type !== 'error' && !hasResults && 
           !results.comparison && !results.timeline && (
            <div className="no-results">
              <p>No results found for your query.</p>
              <p>Try rephrasing or using different keywords.</p>
            </div>
          )}

          {/* Comparative Analysis */}
          {results.comparison && (
            <>
              <h3>Session Comparison</h3>
              <div className="comparison-grid">
                {Object.entries(results.comparison).map(([label, data]) => (
                  <div key={label} className="comparison-card">
                    <h4>{label}</h4>
                    <div className="comparison-metrics">
                      <MetricRow label="Emotional Tone" value={data.metrics.avg_emotional_tone} />
                      <MetricRow label="Analytic Thinking" value={data.metrics.avg_analytic_thinking} />
                      <MetricRow label="Clout" value={data.metrics.avg_clout} />
                      <MetricRow label="Authenticity" value={data.metrics.avg_authenticity} />
                      <MetricRow label="Certainty" value={data.metrics.avg_certainty} />
                    </div>
                    <div className="comparison-stats">
                      <p><strong>Chunks:</strong> {data.total_chunks}</p>
                      <p><strong>Speakers:</strong> {data.unique_speakers}</p>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}

          {/* Timeline */}
          {results.timeline && (
            <>
              <h3>Timeline Analysis</h3>
              <div className="timeline-summary">
                <p><strong>Duration:</strong> {Math.floor(results.summary?.total_duration / 60)} minutes</p>
                <p><strong>Data Points:</strong> {results.summary?.total_chunks}</p>
              </div>
              <div className="timeline-chart">
                {results.timeline.slice(0, 20).map((point, idx) => (
                  <div key={idx} className="timeline-point">
                    <span className="timeline-time">{formatTime(point.time)}</span>
                    <div className="timeline-metrics">
                      <span>Tone: {point.metrics.emotional_tone.toFixed(0)}</span>
                      <span>Analytic: {point.metrics.analytic_thinking.toFixed(0)}</span>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

// Helper component for metric rows in comparison
function MetricRow({ label, value }) {
  return (
    <div className="metric-row">
      <span className="metric-label">{label}:</span>
      <div className="metric-bar">
        <div className="metric-fill" style={{width: `${value}%`}}></div>
        <span className="metric-value">{value.toFixed(0)}</span>
      </div>
    </div>
  );
}

export { RagSearchComponent };