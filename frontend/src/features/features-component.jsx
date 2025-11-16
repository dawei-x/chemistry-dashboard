import { useEffect, useMemo, useState } from "react";
import { FeaturePage } from "./html-pages";
import axios from 'axios';

function AppFeaturesComponent(props) {
  // Check if we're in multi-session Group mode
  const isMultiMode = props.mode === 'Group' && 
                      Array.isArray(props.multiSeries) && 
                      props.multiSeries.length > 0;

  // === ORIGINAL LOGIC FOR INDIVIDUAL/COMPARISON MODES ===
  const svgWidth = 74;
  const svgHeight = 39;
  const [singleFeatures, setSingleFeatures] = useState([]);
  const [featureDescription, setFeatureDescription] = useState(null);
  const [featureHeader, setFeatureHeader] = useState(null);
  const [showFeatureDialog, setShowFeatureDialog] = useState(false);
  
  // === LLM SCORES STATE ===
  const [llmMetricsMap, setLlmMetricsMap] = useState({}); // Map of session_device_id -> LLM metrics
  const [llmLoading, setLlmLoading] = useState(false);
  const [selectedLlmExplanation, setSelectedLlmExplanation] = useState(null);
  const [showLlmExplanationDialog, setShowLlmExplanationDialog] = useState(false);

  // Original updateGraphs for single-transcript mode
  const updateGraphs = () => {
    const valueArrays = [
      { name: 'Emotional tone', values: [] },
      { name: 'Analytic thinking', values: [] },
      { name: 'Clout', values: [] },
      { name: 'Authenticity', values: [] },
      { name: 'Confusion', values: [] }
    ];
    
    props.transcripts.map(t => {
      valueArrays[0].values.push(t.emotional_tone_value);
      valueArrays[1].values.push(t.analytic_thinking_value);
      valueArrays[2].values.push(t.clout_value);
      valueArrays[3].values.push(t.authenticity_value);
      valueArrays[4].values.push(t.certainty_value);
    });
    
    for (const valueArray of valueArrays) {
      const length = valueArray.values.length;
      const average = valueArray.values.reduce((sum, current) => sum + current, 0) / ((length > 0) ? length : 1);
      const last = (length > 0) ? valueArray.values[length - 1] : 0;
      const trend = (last > average) ? 1 : (last === average) ? 0 : -1;
      let path = '';
      for (let i = 0; i < length; i++) {
        const xPos = Math.round(((i + 1) / length) * svgWidth);
        const yPos = svgHeight - Math.round((valueArray.values[i] / 100) * svgHeight);
        path += (i === 0) ? 'M' : 'L';
        path += `${xPos} ${yPos} `;
      }
      valueArray['average'] = average;
      valueArray['last'] = last;
      valueArray['trend'] = trend;
      valueArray['path'] = path;
    }
    setSingleFeatures(valueArrays);
  };

  useEffect(() => {
    if (!isMultiMode) {
      updateGraphs();
    }
  }, [props.transcripts, isMultiMode]);

  // === FETCH LLM METRICS ===
  const fetchLlmMetrics = async (sessionDeviceIds) => {
    try {
      const promises = sessionDeviceIds.map(async (deviceId) => {
        // Fix the ID format - extract just the session_device_id
        const actualId = String(deviceId).includes(':') ? String(deviceId).split(':')[1] : deviceId;
        
        try {
          const response = await axios.get(`/api/v1/session-devices/${actualId}/llm-metrics`);
          if (response.data) {
            return { deviceId: String(deviceId), actualId, metrics: response.data };
          }
        } catch (error) {
          console.log(`No LLM metrics for device ${actualId} yet`);
          return { deviceId: String(deviceId), actualId, metrics: null };
        }
      });
      
      const results = await Promise.all(promises);
      const metricsMap = {};
      results.forEach(({ deviceId, actualId, metrics }) => {
        if (metrics) {
          // Store by both the original deviceId and actualId for compatibility
          metricsMap[deviceId] = metrics;
          metricsMap[actualId] = metrics;
        }
      });
      setLlmMetricsMap(metricsMap);
    } catch (error) {
      console.error('Error fetching LLM metrics:', error);
    }
  };

  // Fetch LLM metrics when devices change
  useEffect(() => {
    if (isMultiMode && props.multiSeries) {
      const deviceIds = props.multiSeries.map(d => d.id);
      fetchLlmMetrics(deviceIds);
    } else if (!isMultiMode && props.session && props.sessionDevice) {
      // For single session mode, fetch for current session_device
      if (props.sessionDevice?.id) {
        fetchLlmMetrics([props.sessionDevice.id]);
      }
    }
  }, [props.multiSeries, isMultiMode, props.sessionDevice]);

  // === NEW MULTI-SESSION LOGIC FOR GROUP MODE ===
  const spec = useMemo(
    () => [
      { name: "Emotional tone", key: "emotional_tone_value", llmKey: "emotional_tone_score", llmExplanationKey: "emotional_tone_explanation" },
      { name: "Analytic thinking", key: "analytic_thinking_value", llmKey: "analytic_thinking_score", llmExplanationKey: "analytic_thinking_explanation" },
      { name: "Clout", key: "clout_value", llmKey: "clout_score", llmExplanationKey: "clout_explanation" },
      { name: "Authenticity", key: "authenticity_value", llmKey: "authenticity_score", llmExplanationKey: "authenticity_explanation" },
      { name: "Confusion", key: "certainty_value", llmKey: "certainty_score", llmExplanationKey: "certainty_explanation" },
    ],
    []
  );

  // Normalize to devices array (only for multi-mode)
  const devices = useMemo(() => {
    if (!isMultiMode) return [];
    
    const src = props.multiSeries;
    if (src && src.length) {
      return src.map((d) => ({
        id: String(d.id),
        label: d.label ?? String(d.id),
        transcripts: Array.isArray(d.transcripts) ? d.transcripts : [],
      }));
    }

    // fallback: single-session from transcripts
    const t = Array.isArray(props.transcripts) ? props.transcripts : [];
    return [{ id: "current", label: "Current session", transcripts: t }];
  }, [props.multiSeries, props.transcripts, isMultiMode]);

  // Compute features with per-device series (only for multi-mode)
  const multiFeatures = useMemo(() => {
    if (!isMultiMode) return [];
    
    const W = 120, H = 36;

    const pathFrom = (values) => {
      const n = values.length;
      if (!n) return "";
      const x = (i) => (n === 1 ? W : i * (W / (n - 1)));
      const y = (v) => H - Math.max(0, Math.min(100, Number(v) || 0)) * (H / 100);
      let d = `M ${x(0)} ${y(values[0])}`;
      for (let i = 1; i < n; i++) d += ` L ${x(i)} ${y(values[i])}`;
      return d;
    };

    const summarize = (arr) => {
      const n = arr.length || 1;
      const sum = arr.reduce((a, v) => a + (Number.isFinite(v) ? v : 0), 0);
      const avg = sum / n;
      const last = arr.length ? arr[arr.length - 1] : 0;
      const trend = last > avg ? 1 : last < avg ? -1 : 0;
      return { average: avg, last, trend };
    };

    return spec.map(({ name, key, llmKey, llmExplanationKey }) => {
      const series = devices.map((dev) => {
        const values = dev.transcripts.map((t) => Number(t?.[key] ?? 0));
        const { average, last, trend } = summarize(values);
        return {
          deviceId: dev.id,
          deviceLabel: dev.label,
          values,
          average,
          last,
          trend,
          path: pathFrom(values),
        };
      });

      // Add LLM series data
      const llmSeries = devices.map((dev) => {
        const llmMetrics = llmMetricsMap[dev.id];
        const score = llmMetrics?.[llmKey] ?? 0;
        const explanation = llmMetrics?.[llmExplanationKey] ?? '';
        return {
          deviceId: dev.id,
          deviceLabel: dev.label,
          score: score,
          explanation: explanation
        };
      });

      return { name, series, llmSeries };
    });
  }, [devices, spec, isMultiMode, llmMetricsMap]);

  // For single mode, enhance features with LLM scores
  const enhancedSingleFeatures = useMemo(() => {
    if (isMultiMode) return [];
    
    const currentDeviceId = props.sessionDevice?.id;
    const llmMetrics = currentDeviceId ? llmMetricsMap[currentDeviceId] : null;
    
    return singleFeatures.map((feature, idx) => {
      const llmKey = spec[idx]?.llmKey;
      const llmExplanationKey = spec[idx]?.llmExplanationKey;
      const llmScore = llmMetrics?.[llmKey] ?? 0;
      const llmExplanation = llmMetrics?.[llmExplanationKey] ?? '';
      
      return {
        ...feature,
        llmScore,
        llmExplanation
      };
    });
  }, [singleFeatures, isMultiMode, spec, llmMetricsMap, props.sessionDevice]);

  // Info dialog handlers
  const getInfo = (name) => {
    const map = {
      "Emotional tone": "Scores above 50 indicate positive tone; below 50 negative.",
      "Analytic thinking": "Scores above 50 indicate analytic thinking; below 50 narrative.",
      Clout: "Higher scores suggest confidence/leadership.",
      Authenticity: "Higher scores suggest more honesty/authenticity.",
      Confusion: "Your pipeline labels this 'certainty'. Higher = more certain.",
    };
    setFeatureDescription(map[name] || null);
    setFeatureHeader(name);
    setShowFeatureDialog(true);
  };
  
  const closeDialog = () => {
    setShowFeatureDialog(false);
    setShowLlmExplanationDialog(false);
  };

  // Show LLM explanation handler
  const showLlmExplanation = (metricName, explanation) => {
    setSelectedLlmExplanation({
      metric: metricName,
      explanation: explanation
    });
    setShowLlmExplanationDialog(true);
  };

  // Choose features based on mode
  const features = isMultiMode ? multiFeatures : enhancedSingleFeatures;
  const optionList = isMultiMode ? props.deviceOptions : undefined;
  const selectedIds = isMultiMode ? props.selectedDeviceIds : undefined;
  const selectionHandler = isMultiMode ? props.onDeviceSelectionChange : undefined;

  // Refresh LLM scores function - FIXED to actually work
  const refreshLlmScores = async () => {
    setLlmLoading(true);
    
    try {
      let deviceIds = [];
      
      if (isMultiMode && props.multiSeries) {
        deviceIds = props.multiSeries.map(d => {
          const id = String(d.id);
          return id.includes(':') ? id.split(':')[1] : id;
        });
      } else if (!isMultiMode && props.sessionDevice?.id) {
        const id = String(props.sessionDevice.id);
        deviceIds = [id.includes(':') ? id.split(':')[1] : id];
      }
      
      // Generate new scores for each device
      for (const deviceId of deviceIds) {
        try {
          await axios.post(`/api/v1/session-devices/${deviceId}/llm-metrics`);
          console.log(`Generated LLM scores for device ${deviceId}`);
        } catch (error) {
          console.error(`Failed to refresh LLM scores for device ${deviceId}:`, error);
        }
      }
      
      // Re-fetch all scores after generation
      if (isMultiMode && props.multiSeries) {
        await fetchLlmMetrics(props.multiSeries.map(d => d.id));
      } else if (!isMultiMode && props.sessionDevice?.id) {
        await fetchLlmMetrics([props.sessionDevice.id]);
      }
      
    } catch (error) {
      console.error('Error refreshing LLM scores:', error);
    } finally {
      setLlmLoading(false);
    }
  };

  return (
    <FeaturePage
      features={features}
      showFeatures={props.showFeatures}
      deviceOptions={optionList}
      selectedDeviceIds={selectedIds}
      onDeviceSelectionChange={selectionHandler}
      currentSessionDeviceId={props.currentSessionDeviceId}
      featureHeader={featureHeader}
      featureDescription={featureDescription}
      showFeatureDialog={showFeatureDialog}
      closeDialog={closeDialog}
      isMulti={isMultiMode}
      getInfo={getInfo}
      mode={props.mode}
      refreshLlmScores={refreshLlmScores}
      llmLoading={llmLoading}
      showLlmExplanation={showLlmExplanation}
      selectedLlmExplanation={selectedLlmExplanation}
      showLlmExplanationDialog={showLlmExplanationDialog}
    />
  );
}

export { AppFeaturesComponent };