import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import Cytoscape from 'cytoscape';
import dagre from 'cytoscape-dagre';
import cola from 'cytoscape-cola';
import style from './concept-map.module.css';
import { SpeakerPanel } from '../speaker-panel/SpeakerPanel';

// Note: Real-time WebSocket updates removed - concepts are now generated post-discussion

// Register layouts
Cytoscape.use(dagre);
Cytoscape.use(cola);

// Semantic zoom thresholds - Google Maps style detail levels
const ZOOM_THRESHOLDS = {
  CLUSTERS_ONLY: 0.6,      // Below this: only cluster nodes visible
  CLUSTERS_EXPANDED: 1.0,  // Below this: clusters with labels, concepts fading in
  FULL_DETAIL: 1.5         // Above this: concepts prominent, clusters faded
};

// Animation timing constants
const ANIMATION = {
  OPACITY_TRANSITION: 200,  // ms for opacity fades
  STAGGER_DELAY: 30,        // ms between each node in staggered animation
  NODE_EXPAND: 400,         // ms for node position animation
  EDGE_FADE: 300            // ms for edge fade-in
};

function ConceptMapView({ sessionId, sessionDeviceId }) {
  const cyRef = useRef(null);
  const containerRef = useRef(null);
  const expandedContainerRef = useRef(null);
  const mountedRef = useRef(true);
  const timeoutsRef = useRef([]);
  const initializingRef = useRef(false);

  const [transcriptPanel, setTranscriptPanel] = useState(null);
  const [panelTranscripts, setPanelTranscripts] = useState([]);
  const [transcriptsLoading, setTranscriptsLoading] = useState(false);
  const [transcriptsError, setTranscriptsError] = useState(null);

  // Hover tooltip state
  const [tooltip, setTooltip] = useState({ visible: false, x: 0, y: 0, data: null });

  // State management
  const [conceptData, setConceptData] = useState({ nodes: [], edges: [] });
  const [clusters, setClusters] = useState([]);
  const [viewMode, setViewMode] = useState('clustered'); // 'clustered' | 'full'
  const [isLoading, setIsLoading] = useState(true);
  const [isExpanded, setIsExpanded] = useState(false);
  const [generationStatus, setGenerationStatus] = useState('pending'); // 'pending' | 'processing' | 'completed' | 'failed'
  const [displayData, setDisplayData] = useState({
    nodeCount: 0,
    edgeCount: 0,
    discourseType: 'exploratory'
  });

  const [selectedSpeakers, setSelectedSpeakers] = useState([]); // Empty = show all
  const [showSpeakerPanel, setShowSpeakerPanel] = useState(true);

  // Semantic zoom state
  const [viewLocked, setViewLocked] = useState(false);  // Lock view toggle - prevents auto-switching
  const [currentZoomLevel, setCurrentZoomLevel] = useState('CLUSTERS_ONLY');
  const positionCacheRef = useRef(new Map());  // Cache pre-computed node positions
  const pendingZoomRef = useRef(null);  // requestAnimationFrame debouncing

  // Node colors configuration - Modern Tailwind-inspired palette
  const nodeColors = useMemo(() => ({
    question: '#EF4444',    // Red-500
    idea: '#3B82F6',        // Blue-500
    example: '#10B981',     // Emerald-500
    uncertainty: '#F59E0B', // Amber-500
    action: '#8B5CF6',      // Violet-500
    goal: '#0EA5E9',        // Sky-500
    problem: '#DC2626',     // Red-600
    solution: '#059669',    // Emerald-600
    elaboration: '#475569', // Slate-600
    synthesis: '#A855F7',   // Purple-500
    challenge: '#EA580C',   // Orange-600
    constraint: '#64748B',  // Slate-500
    default: '#6B7280'      // Gray-500
  }), []);

  // Color helper
  const darkenColor = useCallback((color, factor) => {
    const num = parseInt(color.replace("#",""), 16);
    const amt = Math.round(2.55 * factor * 100);
    const R = (num >> 16) - amt;
    const G = (num >> 8 & 0x00FF) - amt;
    const B = (num & 0x0000FF) - amt;
    return "#" + (0x1000000 + (R<255?R<1?0:R:255)*0x10000 + (G<255?G<1?0:G:255)*0x100 + (B<255?B<1?0:B:255)).toString(16).slice(1);
  }, []);

  // Format edge labels
  const formatEdgeLabel = useCallback((type) => {
    if (!type) return '';
    return type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  }, []);

  // Determine semantic zoom level from numeric zoom value
  const getZoomLevel = useCallback((zoom) => {
    if (zoom < ZOOM_THRESHOLDS.CLUSTERS_ONLY) return 'CLUSTERS_ONLY';
    if (zoom < ZOOM_THRESHOLDS.CLUSTERS_EXPANDED) return 'CLUSTERS_LABELED';
    if (zoom < ZOOM_THRESHOLDS.FULL_DETAIL) return 'EXPANDED';
    return 'FULL_DETAIL';
  }, []);

  // Calculate smooth opacity transitions between zoom levels
  const calculateOpacities = useCallback((zoom) => {
    let clusterBgOpacity, clusterBorderOpacity, clusterTextOpacity;
    let nodeOpacity, intraEdgeOpacity, interEdgeOpacity;

    if (zoom < ZOOM_THRESHOLDS.CLUSTERS_ONLY) {
      // Clusters only - collapsed appearance
      clusterBgOpacity = 0.7;
      clusterBorderOpacity = 0.8;
      clusterTextOpacity = 1;
      nodeOpacity = 0;
      intraEdgeOpacity = 0;
      interEdgeOpacity = 0.8;
    } else if (zoom < ZOOM_THRESHOLDS.CLUSTERS_EXPANDED) {
      // Transition zone: clusters start showing content
      const t = (zoom - ZOOM_THRESHOLDS.CLUSTERS_ONLY) /
                (ZOOM_THRESHOLDS.CLUSTERS_EXPANDED - ZOOM_THRESHOLDS.CLUSTERS_ONLY);
      clusterBgOpacity = 0.7 - (0.4 * t);  // 0.7 -> 0.3
      clusterBorderOpacity = 0.8;
      clusterTextOpacity = 1;
      nodeOpacity = t * 0.6;  // 0 -> 0.6
      intraEdgeOpacity = t * 0.4;
      interEdgeOpacity = 0.8 - (0.3 * t);
    } else if (zoom < ZOOM_THRESHOLDS.FULL_DETAIL) {
      // Expanded view - children becoming more prominent
      const t = (zoom - ZOOM_THRESHOLDS.CLUSTERS_EXPANDED) /
                (ZOOM_THRESHOLDS.FULL_DETAIL - ZOOM_THRESHOLDS.CLUSTERS_EXPANDED);
      clusterBgOpacity = 0.3 - (0.15 * t);  // 0.3 -> 0.15
      clusterBorderOpacity = 0.8 - (0.4 * t);
      clusterTextOpacity = 1 - (0.4 * t);
      nodeOpacity = 0.6 + (0.4 * t);  // 0.6 -> 1.0
      intraEdgeOpacity = 0.4 + (0.6 * t);  // 0.4 -> 1.0
      interEdgeOpacity = 0.5 - (0.3 * t);
    } else {
      // Full detail - nodes prominent, clusters faded
      clusterBgOpacity = 0.1;
      clusterBorderOpacity = 0.3;
      clusterTextOpacity = 0.5;
      nodeOpacity = 1;
      intraEdgeOpacity = 1;
      interEdgeOpacity = 0.2;
    }

    return {
      cluster: { bgOpacity: clusterBgOpacity, borderOpacity: clusterBorderOpacity, textOpacity: clusterTextOpacity },
      node: { opacity: nodeOpacity },
      intraEdge: { opacity: intraEdgeOpacity },
      interEdge: { opacity: interEdgeOpacity }
    };
  }, []);

  // Handle semantic zoom - update opacities based on zoom level
  // Uses requestAnimationFrame for smooth debouncing
  const handleSemanticZoom = useCallback((cy) => {
    if (!cy || viewLocked) return;  // Skip if view is locked

    // Cancel any pending zoom operation for smoother performance
    if (pendingZoomRef.current) {
      cancelAnimationFrame(pendingZoomRef.current);
    }

    // Use requestAnimationFrame for smooth debouncing
    pendingZoomRef.current = requestAnimationFrame(() => {
      const zoom = cy.zoom();

      // Calculate opacity values for current zoom
      const opacities = calculateOpacities(zoom);

      // Determine if we should hide concept nodes completely (for equal cluster sizing)
      const shouldHideNodes = zoom < ZOOM_THRESHOLDS.CLUSTERS_ONLY;

      // Use batch for performance - all updates in single redraw
      // CSS transitions handle the smooth animation
      cy.batch(() => {
        // Update cluster nodes with explicit sizing control
        cy.nodes('[isCluster]').forEach(node => {
          node.data('bgOpacity', opacities.cluster.bgOpacity);
          node.data('borderOpacity', opacities.cluster.borderOpacity);
          node.data('textOpacity', opacities.cluster.textOpacity);

          // Force uniform cluster size when zoomed out (children hidden)
          // This ensures all clusters appear equal size in overview mode
          if (shouldHideNodes) {
            node.style({
              'width': '180px',
              'height': '120px',
              'shape': 'ellipse',
              'text-valign': 'center'
            });
          } else {
            // Let clusters auto-size based on children when expanded
            node.style({
              'width': 'auto',
              'height': 'auto',
              'shape': 'round-rectangle',
              'text-valign': 'top'
            });
          }
        });

        // Update concept nodes - use display:none at low zoom for equal cluster sizes
        cy.nodes('[!isCluster]').forEach(node => {
          node.data('nodeOpacity', opacities.node.opacity);
          // Hide completely at low zoom so compound nodes have equal sizes
          node.style('display', shouldHideNodes ? 'none' : 'element');
        });

        // Update intra-cluster edges
        cy.edges('[!interCluster]').forEach(edge => {
          edge.data('edgeOpacity', opacities.intraEdge.opacity);
          edge.style('display', shouldHideNodes ? 'none' : 'element');
        });

        // Update inter-cluster edges
        cy.edges('[interCluster]').forEach(edge => {
          edge.data('interEdgeOpacity', opacities.interEdge.opacity);
        });
      });

      // Update state for UI indicator
      setCurrentZoomLevel(getZoomLevel(zoom));
    });
  }, [viewLocked, getZoomLevel, calculateOpacities]);

  // Initialize Cytoscape with compound node support
  const initCytoscape = useCallback((container) => {
    if (!container) return null;
    
    const cy = Cytoscape({
      container: container,
      style: [
        // Cluster (parent) node styles - with dynamic opacity for semantic zoom
        {
          selector: 'node[isCluster]',
          style: {
            'shape': 'round-rectangle',
            'background-color': 'data(color)',
            'background-opacity': 'data(bgOpacity)',  // Dynamic opacity
            'border-width': 2,
            'border-color': 'data(borderColor)',
            'border-opacity': 'data(borderOpacity)',  // Dynamic opacity
            'label': 'data(label)',
            'text-valign': 'top',
            'text-halign': 'center',
            'font-size': '16px',
            'font-weight': 'bold',
            'text-opacity': 'data(textOpacity)',  // Dynamic opacity
            'padding': '40px',  // Increased padding for better spacing
            'text-margin-y': 10,
            'min-width': '180px',
            'min-height': '120px',
            'z-index': 1,
            // CSS transitions for smooth opacity AND size changes
            'transition-property': 'background-opacity, border-opacity, text-opacity, width, height',
            'transition-duration': `${ANIMATION.OPACITY_TRANSITION}ms`
          }
        },
        // Collapsed cluster styles (when all collapsed)
        {
          selector: 'node[isCluster][collapsed]',
          style: {
            'shape': 'ellipse',
            'width': '180px',
            'height': '120px',
            'text-valign': 'center',
            'padding': '10px',
            'cursor': 'pointer'
          }
        },
        // Regular node styles - with dynamic opacity for semantic zoom
        {
          selector: 'node[!isCluster]',
          style: {
            'shape': 'round-rectangle',  // Better for text
            'width': '180px',            // Increased from 140px
            'height': 44,                // Increased from 36
            'label': 'data(label)',
            'text-valign': 'center',
            'text-halign': 'center',
            'background-color': 'data(color)',
            'background-opacity': 'data(nodeOpacity)',  // Dynamic opacity
            'border-width': 2,
            'border-color': 'data(borderColor)',
            'border-opacity': 0.6,
            'font-size': '12px',         // Increased from 11px
            'font-weight': 500,
            'color': '#ffffff',
            'text-opacity': 'data(nodeOpacity)',  // Dynamic opacity
            'text-wrap': 'ellipsis',
            'text-max-width': '160px',   // Increased from 125px
            'z-index': 10,
            // Subtle shadow effect via overlay
            'overlay-opacity': 0,
            // CSS transitions for smooth opacity changes
            'transition-property': 'background-opacity, border-opacity, text-opacity, opacity',
            'transition-duration': `${ANIMATION.OPACITY_TRANSITION}ms`
          }
        },
        // Hidden nodes when cluster is collapsed
        {
          selector: 'node[?hidden]',
          style: {
            'display': 'none'
          }
        },
        // Edge styles - with dynamic opacity
        {
          selector: 'edge',
          style: {
            'curve-style': 'bezier',
            'target-arrow-shape': 'triangle-backcurve',  // Softer arrow
            'arrow-scale': 0.8,
            'width': 2,
            'line-color': '#94A3B8',           // Softer gray (Slate-400)
            'target-arrow-color': '#94A3B8',
            'opacity': 'data(edgeOpacity)',    // Dynamic opacity
            'label': 'data(label)',
            'font-size': '11px',               // Increased from 10px
            'font-weight': 500,
            'text-rotation': 'autorotate',
            'text-background-color': '#FFFFFF', // White background for readability
            'text-background-opacity': 0.9,
            'text-background-padding': '3px',
            'text-background-shape': 'roundrectangle',
            'color': '#475569',                // Slate-600 for contrast
            'text-opacity': 'data(edgeOpacity)',
            'z-index': 5,
            // CSS transitions
            'transition-property': 'opacity, text-opacity',
            'transition-duration': `${ANIMATION.OPACITY_TRANSITION}ms`
          }
        },
        // Hidden edges
        {
          selector: 'edge[?hidden]',
          style: {
            'display': 'none'
          }
        },
        // Inter-cluster edges - with dynamic opacity
        {
          selector: 'edge[interCluster]',
          style: {
            'line-color': '#3498db',
            'target-arrow-color': '#3498db',
            'width': 3,
            'line-style': 'dashed',
            'opacity': 'data(interEdgeOpacity)',  // Dynamic opacity
            'transition-property': 'opacity',
            'transition-duration': `${ANIMATION.OPACITY_TRANSITION}ms`
          }
        }
      ],
      layout: {
        name: 'preset'
      },
      minZoom: 0.3,
      maxZoom: 3,
      wheelSensitivity: 0.2,
      userZoomingEnabled: true,
      userPanningEnabled: true,
      autoungrabify: false
    });

    // NOTE: Event handlers are NOT registered here
    // They are registered in renderClusteredView and renderFullView
    // to ensure proper cleanup on view mode switch

    return cy;
  }, []);

  // Load concept data with generation status handling
  useEffect(() => {
    if (!sessionDeviceId) return;

    console.log('SessionDeviceId being used:', sessionDeviceId);

    const loadConceptData = () => {
      setIsLoading(true);

      // Load concept data (includes generation_status)
      fetch(`/api/v1/concepts/${sessionDeviceId}`)
        .then(res => res.json())
        .then(data => {
          // Update generation status from API response
          if (data.generation_status) {
            setGenerationStatus(data.generation_status);
          }

          if (data && data.nodes && data.nodes.length > 0) {
            setConceptData(data);
            setDisplayData({
              nodeCount: data.nodes.length,
              edgeCount: data.edges.length,
              discourseType: data.discourse_type || 'exploratory'
            });

            // Load or create clusters
            return fetch(`/api/v1/concepts/${sessionDeviceId}/clusters`);
          } else if (data.generation_status === 'processing') {
            // No data yet but generation is in progress
            return null;
          }
          throw new Error('No concept data');
        })
        .then(res => res ? res.json() : null)
        .then(clusterData => {
          if (!clusterData) return;

          console.log('Clusters loaded:', clusterData);
          if (clusterData.clusters && clusterData.clusters.length > 0) {
            setClusters(clusterData.clusters);
          } else if (conceptData.nodes.length > 0) {
            // Trigger clustering if we have nodes but no clusters
            return fetch(`/api/v1/concepts/${sessionDeviceId}/cluster`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ method: 'semantic' })
            })
            .then(() => fetch(`/api/v1/concepts/${sessionDeviceId}/clusters`))
            .then(res => res.json())
            .then(newClusterData => {
              setClusters(newClusterData.clusters || []);
            });
          }
        })
        .catch(err => {
          console.error('Failed to load data:', err);
        })
        .finally(() => {
          setIsLoading(false);
        });
    };

    // Initial load
    loadConceptData();

    // Poll for updates if generation is in progress
    let pollInterval = null;
    if (generationStatus === 'processing' || generationStatus === 'pending') {
      pollInterval = setInterval(() => {
        fetch(`/api/v1/concepts/${sessionDeviceId}`)
          .then(res => res.json())
          .then(data => {
            if (data.generation_status) {
              setGenerationStatus(data.generation_status);

              // If generation completed, reload full data
              if (data.generation_status === 'completed' && data.nodes && data.nodes.length > 0) {
                loadConceptData();
                clearInterval(pollInterval);
              }
            }
          })
          .catch(err => console.error('Polling error:', err));
      }, 5000); // Poll every 5 seconds
    }

    return () => {
      if (pollInterval) clearInterval(pollInterval);
    };
  }, [sessionDeviceId, generationStatus]);

  // Note: Real-time WebSocket updates removed - concepts are now generated post-discussion
  // The polling mechanism above handles checking for generation completion

  // Render clustered view with semantic zoom support
  // All elements are pre-rendered, visibility controlled via opacity
  const renderClusteredView = useCallback((cy) => {
    const elements = [];
    // Pass cluster sizes for dynamic spacing calculation
    const clusterSizes = clusters.map(c => c.nodes?.length || c.node_count || 0);
    const clusterPositions = calculateClusterPositions(clusters.length, clusterSizes);

    // Get initial opacities based on current zoom (default to 1.0 for initial render)
    const initialZoom = cy.zoom() || 1.0;
    const initialOpacities = calculateOpacities(initialZoom);

    // Determine initial cluster sizing based on zoom
    const shouldStartCollapsed = initialZoom < ZOOM_THRESHOLDS.CLUSTERS_ONLY;

    clusters.forEach((cluster, index) => {
      const clusterColor = getClusterColor(index);

      // Add cluster parent node with opacity data
      // Start with uniform size if zoomed out, auto-size if zoomed in
      elements.push({
        data: {
          id: `cluster_${cluster.id}`,
          label: cluster.name || `Cluster ${index + 1}`,
          isCluster: true,
          color: clusterColor,
          borderColor: darkenColor(clusterColor, 0.3),
          nodeCount: cluster.node_count || cluster.nodes?.length || 0,
          summary: cluster.summary,
          // Dynamic opacity values for semantic zoom
          bgOpacity: initialOpacities.cluster.bgOpacity,
          borderOpacity: initialOpacities.cluster.borderOpacity,
          textOpacity: initialOpacities.cluster.textOpacity
        },
        position: clusterPositions[index],
        // Apply uniform sizing when starting zoomed out
        style: shouldStartCollapsed ? {
          'width': '180px',
          'height': '120px',
          'shape': 'ellipse',
          'text-valign': 'center'
        } : {}
      });

      // ALWAYS add individual nodes (visibility controlled by opacity)
      if (cluster.nodes) {
        const nodePositions = calculateNodePositions(
          cluster.nodes.length,
          clusterPositions[index]
        );

        // Cache positions for animated expansion
        const clusterPositionCache = new Map();

        cluster.nodes.forEach((node, nodeIndex) => {
          const nodeColor = nodeColors[node.type] || nodeColors.default;
          const position = nodePositions[nodeIndex];

          // Cache the position
          clusterPositionCache.set(node.id, position);

          // Determine if nodes should be hidden initially (for equal cluster sizing)
          const shouldHideInitially = initialZoom < ZOOM_THRESHOLDS.CLUSTERS_ONLY;

          elements.push({
            data: {
              id: node.id,
              parent: `cluster_${cluster.id}`,
              label: node.text,
              type: node.type,
              color: nodeColor,
              borderColor: darkenColor(nodeColor, 0.2),
              timestamp: node.timestamp,
              clusterId: cluster.id,
              speaker_id: node.speaker_id,
              speaker_alias: node.speaker_alias,
              // Dynamic opacity for semantic zoom
              nodeOpacity: initialOpacities.node.opacity
            },
            position: position,
            // Hide initially at low zoom for equal cluster sizes
            style: shouldHideInitially ? { display: 'none' } : {}
          });
        });

        // Store in position cache
        positionCacheRef.current.set(cluster.id, clusterPositionCache);

        // Add edges within cluster
        if (cluster.edges) {
          // Determine if edges should be hidden initially
          const shouldHideInitially = initialZoom < ZOOM_THRESHOLDS.CLUSTERS_ONLY;

          cluster.edges.forEach(edge => {
            elements.push({
              data: {
                id: edge.id,
                source: edge.source,
                target: edge.target,
                label: formatEdgeLabel(edge.type),
                // Dynamic opacity for semantic zoom
                edgeOpacity: initialOpacities.intraEdge.opacity
              },
              // Hide initially at low zoom for equal cluster sizes
              style: shouldHideInitially ? { display: 'none' } : {}
            });
          });
        }
      }
    });

    // Add inter-cluster edges
    const interClusterEdges = findInterClusterEdges();
    interClusterEdges.forEach((edge, index) => {
      elements.push({
        data: {
          id: `inter_edge_${index}`,
          source: `cluster_${edge.sourceCluster}`,
          target: `cluster_${edge.targetCluster}`,
          interCluster: true,
          label: `${edge.count} connection${edge.count > 1 ? 's' : ''}`,
          // Dynamic opacity for semantic zoom
          interEdgeOpacity: initialOpacities.interEdge.opacity
        }
      });
    });

    cy.add(elements);

    // Register zoom event handler for semantic zoom
    cy.off('zoom');
    cy.on('zoom', () => {
      handleSemanticZoom(cy);
    });

    // Add click handler for cluster nodes - zoom into cluster
    cy.off('tap');
    cy.on('tap', 'node[isCluster]', (evt) => {
      const node = evt.target;
      // Animate zoom into the cluster
      cy.animate({
        zoom: ZOOM_THRESHOLDS.FULL_DETAIL + 0.2,
        center: { eles: node },
        duration: 300,
        easing: 'ease-out-cubic'
      });
    });

    // Click handler for concept nodes to show transcripts
    cy.on('tap', 'node[!isCluster]', function(evt) {
      const node = evt.target;
      const timestamp = node.data('timestamp');

      setTranscriptPanel({
        nodeText: node.data('label'),
        timestamp: timestamp || 0,
        speakerAlias: node.data('speaker_alias')
      });

      // Handle missing or invalid timestamps
      if (timestamp === undefined || timestamp === null || timestamp <= 0) {
        console.warn('Concept has no valid timestamp:', node.data('label'));
        setTranscriptsLoading(false);
        setTranscriptsError('This concept was synthesized from the discussion and has no specific source.');
        setPanelTranscripts([]);
        return;
      }

      // Start loading
      setTranscriptsLoading(true);
      setTranscriptsError(null);

      // Use sessionDeviceId prop instead of parsing from node ID
      // This works for both real-time (node_123:456_0) and post-discussion (node_42_0) formats
      // Use max(0, timestamp - 15) to avoid negative timestamps
      const adjustedTimestamp = Math.max(0, Math.floor(timestamp) - 15);
      const url = `/api/v1/concepts/${sessionDeviceId}/transcripts/${adjustedTimestamp}`;
      fetch(url)
        .then(res => {
          if (!res.ok) {
            throw new Error(`HTTP ${res.status}`);
          }
          return res.json();
        })
        .then(data => {
          setPanelTranscripts(Array.isArray(data) ? data : []);
          setTranscriptsLoading(false);
        })
        .catch(err => {
          console.error('Failed to load transcripts:', err);
          setPanelTranscripts([]);
          setTranscriptsLoading(false);
          setTranscriptsError('Failed to load source transcripts.');
        });
    });

    // Hover tooltip handlers - only show when zoomed in enough to see concepts
    cy.on('mouseover', 'node[!isCluster]', function(evt) {
      const currentZoom = cy.zoom();
      // Only show tooltip when concepts are visible (above CLUSTERS_ONLY threshold)
      if (currentZoom < ZOOM_THRESHOLDS.CLUSTERS_EXPANDED) {
        return; // Don't show tooltip when concepts are hidden/faded
      }

      const node = evt.target;
      const renderedPos = node.renderedPosition();
      const container = cy.container().getBoundingClientRect();

      setTooltip({
        visible: true,
        x: container.left + renderedPos.x + 20,
        y: container.top + renderedPos.y - 10,
        data: {
          label: node.data('label'),
          type: node.data('type'),
          timestamp: node.data('timestamp'),
          color: node.data('color'),
          speakerId: node.data('speaker_id'),
          speakerAlias: node.data('speaker_alias')
        }
      });
    });

    cy.on('mouseout', 'node[!isCluster]', function() {
      setTooltip(prev => ({ ...prev, visible: false }));
    });

    // Run layout once - positions will be cached, no recalculation on zoom
    cy.layout({
      name: 'dagre',
      rankDir: 'TB',
      padding: 80,          // Increased from 50 for better cluster separation
      spacingFactor: 1.5,   // Increased from 1.2 for more breathing room
      nodeSep: 60,          // Horizontal separation between nodes
      rankSep: 80,          // Vertical separation between ranks
      animate: true,
      animationDuration: 500,
      fit: true
    }).run();

    // Apply initial opacity based on zoom level after layout
    setTimeout(() => {
      handleSemanticZoom(cy);
    }, 100);
  }, [clusters, nodeColors, darkenColor, formatEdgeLabel, calculateOpacities, handleSemanticZoom, setTranscriptPanel, setPanelTranscripts, sessionDeviceId]);

  // Render full view - all nodes visible with full opacity
  const renderFullView = useCallback((cy) => {
    const elements = [];

    // Add all nodes with full opacity (full detail mode)
    conceptData.nodes.forEach(node => {
      const nodeColor = nodeColors[node.type] || nodeColors.default;
      elements.push({
        data: {
          id: node.id,
          label: node.text,
          type: node.type,
          color: nodeColor,
          borderColor: darkenColor(nodeColor, 0.2),
          speaker_id: node.speaker_id,
          speaker_alias: node.speaker_alias,
          timestamp: node.timestamp,
          // Full opacity in full view mode
          nodeOpacity: 1
        }
      });
    });

    // Add all edges with full opacity
    conceptData.edges.forEach(edge => {
      elements.push({
        data: {
          id: edge.id,
          source: edge.source,
          target: edge.target,
          label: formatEdgeLabel(edge.type),
          edgeOpacity: 1
        }
      });
    });

    cy.add(elements);

    // Run layout
    cy.layout({
      name: 'dagre',
      rankDir: 'TB',
      padding: 40,
      spacingFactor: 1.5,
      animate: true,
      animationDuration: 500
    }).run();

    // Click handler for nodes to show transcripts
    cy.off('tap', 'node');
    cy.on('tap', 'node', function(evt) {
      const node = evt.target;
      const timestamp = node.data('timestamp');

      setTranscriptPanel({
        nodeText: node.data('label'),
        timestamp: timestamp || 0,
        speakerAlias: node.data('speaker_alias')
      });

      // Handle missing or invalid timestamps
      if (timestamp === undefined || timestamp === null || timestamp <= 0) {
        console.warn('Concept has no valid timestamp:', node.data('label'));
        setTranscriptsLoading(false);
        setTranscriptsError('This concept was synthesized from the discussion and has no specific source.');
        setPanelTranscripts([]);
        return;
      }

      // Start loading
      setTranscriptsLoading(true);
      setTranscriptsError(null);

      // Use sessionDeviceId prop instead of parsing from node ID
      // This works for both real-time (node_123:456_0) and post-discussion (node_42_0) formats
      // Use max(0, timestamp - 20) to avoid negative timestamps
      const adjustedTimestamp = Math.max(0, Math.floor(timestamp) - 20);
      fetch(`/api/v1/concepts/${sessionDeviceId}/transcripts/${adjustedTimestamp}`)
        .then(res => {
          if (!res.ok) {
            throw new Error(`HTTP ${res.status}`);
          }
          return res.json();
        })
        .then(data => {
          setPanelTranscripts(Array.isArray(data) ? data : []);
          setTranscriptsLoading(false);
        })
        .catch(err => {
          console.error('Failed to load transcripts:', err);
          setPanelTranscripts([]);
          setTranscriptsLoading(false);
          setTranscriptsError('Failed to load source transcripts.');
        });
    });

    // Hover tooltip handlers
    cy.on('mouseover', 'node', function(evt) {
      const node = evt.target;
      const renderedPos = node.renderedPosition();
      const container = cy.container().getBoundingClientRect();

      setTooltip({
        visible: true,
        x: container.left + renderedPos.x + 20,
        y: container.top + renderedPos.y - 10,
        data: {
          label: node.data('label'),
          type: node.data('type'),
          timestamp: node.data('timestamp'),
          color: node.data('color'),
          speakerId: node.data('speaker_id'),
          speakerAlias: node.data('speaker_alias')
        }
      });
    });

    cy.on('mouseout', 'node', function() {
      setTooltip(prev => ({ ...prev, visible: false }));
    });
  }, [conceptData, nodeColors, darkenColor, formatEdgeLabel, setTranscriptPanel, setPanelTranscripts, sessionDeviceId]);

  // Build and render the graph with semantic zoom support
  useEffect(() => {
    const container = isExpanded ? expandedContainerRef.current : containerRef.current;
    if (!container || !conceptData.nodes || conceptData.nodes.length === 0) return;

    // Initialize Cytoscape if not already done
    if (!cyRef.current) {
      cyRef.current = initCytoscape(container);
    } else {
      cyRef.current.mount(container);
    }

    const cy = cyRef.current;

    // CRITICAL: Clear ALL event handlers before mode switch
    // This prevents semantic zoom handler from clustered view affecting full view
    cy.removeAllListeners();

    // Clear existing elements
    cy.elements().remove();

    // Render based on view mode
    // Clustered view with semantic zoom OR full view
    if (viewMode === 'clustered' && clusters.length > 0) {
      renderClusteredView(cy);
    } else {
      renderFullView(cy);
    }

    // Fit to viewport after layout completes
    setTimeout(() => {
      cy.fit(50);
      cy.center();
    }, 600);  // Wait for layout animation to complete

    // Cleanup on unmount
    return () => {
      if (pendingZoomRef.current) {
        cancelAnimationFrame(pendingZoomRef.current);
      }
    };
  }, [conceptData, clusters, viewMode, isExpanded, initCytoscape, renderClusteredView, renderFullView]);

  const applySpeakerHighlighting = useCallback(() => {
  if (!cyRef.current || viewMode !== 'full') return;
  
  const cy = cyRef.current;

  // Debug: Check what we're working with
  const firstNode = cy.nodes()[0];
  if (firstNode) {
    console.log('Debug - First node speaker_id:', firstNode.data('speaker_id'), 'type:', typeof firstNode.data('speaker_id'));
    console.log('Debug - selectedSpeakers:', selectedSpeakers);
    console.log('Debug - selectedSpeakers types:', selectedSpeakers.map(s => typeof s));
  }
  
  if (selectedSpeakers.length === 0) {
    cy.nodes().style({'opacity': 1, 'border-width': 2});
    cy.edges().style('opacity', 1);
    return;
  }
  
  cy.nodes().forEach(node => {
    const speakerId = node.data('speaker_id');
    // Convert to string for consistent comparison
    const isSelected = selectedSpeakers.includes(String(speakerId)) || 
                      selectedSpeakers.includes(Number(speakerId)) ||
                      (speakerId === null && selectedSpeakers.includes('Unknown'));
    
    node.style({
      'opacity': isSelected ? 1 : 0.3,
      'border-width': isSelected ? 4 : 2
    });
  });
  
  // Update edges too
  cy.edges().forEach(edge => {
    const sourceId = cy.getElementById(edge.data('source')).data('speaker_id');
    const targetId = cy.getElementById(edge.data('target')).data('speaker_id');
    const isRelevant = selectedSpeakers.includes(String(sourceId)) || 
                      selectedSpeakers.includes(String(targetId)) ||
                      selectedSpeakers.includes(Number(sourceId)) || 
                      selectedSpeakers.includes(Number(targetId));
    
    edge.style('opacity', isRelevant ? 1 : 0.2);
  });
}, [selectedSpeakers, viewMode]);

useEffect(() => {
  applySpeakerHighlighting();
  }, [selectedSpeakers, applySpeakerHighlighting]);

  // Helper: Calculate cluster positions in a circle with dynamic spacing
  const calculateClusterPositions = (count, clusterSizes = []) => {
    const positions = [];
    const center = { x: 400, y: 300 };

    // Dynamic radius based on cluster count and sizes
    // More clusters or larger clusters need more space to avoid overlap when expanded
    const maxClusterSize = clusterSizes.length > 0 ? Math.max(...clusterSizes, 5) : 5;
    const baseRadius = 200;
    const perClusterSpace = 220; // Space needed per cluster when expanded
    const circumferenceNeeded = count * perClusterSpace;
    const radiusFromCircumference = circumferenceNeeded / (2 * Math.PI);

    // Add extra space for larger clusters
    const sizeBonus = Math.sqrt(maxClusterSize) * 25;

    const radius = Math.max(baseRadius, radiusFromCircumference + sizeBonus);

    for (let i = 0; i < count; i++) {
      const angle = (2 * Math.PI * i) / count - Math.PI / 2;
      positions.push({
        x: center.x + radius * Math.cos(angle),
        y: center.y + radius * Math.sin(angle)
      });
    }
    return positions;
  };

  // Helper: Calculate node positions within a cluster
  const calculateNodePositions = (count, clusterCenter) => {
    const positions = [];

    if (count === 0) return positions;

    // Single node - center it
    if (count === 1) {
      return [{ x: clusterCenter.x, y: clusterCenter.y }];
    }

    const cols = Math.ceil(Math.sqrt(count));
    const rows = Math.ceil(count / cols);
    // Increased spacing to prevent overlap (was 100)
    const spacing = 130;

    for (let i = 0; i < count; i++) {
      const row = Math.floor(i / cols);
      const col = i % cols;
      // Center the grid properly
      const offsetX = (col - (cols - 1) / 2) * spacing;
      const offsetY = (row - (rows - 1) / 2) * spacing;
      positions.push({
        x: clusterCenter.x + offsetX,
        y: clusterCenter.y + offsetY
      });
    }
    return positions;
  };

  // Helper: Get cluster color - Softer, more muted palette
  const getClusterColor = (index) => {
    const clusterColors = [
      '#93C5FD',   // Blue-300
      '#FDA4AF',   // Rose-300
      '#86EFAC',   // Green-300
      '#FCD34D',   // Amber-300
      '#C4B5FD',   // Violet-300
      '#67E8F9',   // Cyan-300
      '#FDBA74',   // Orange-300
    ];
    return clusterColors[index % clusterColors.length];
  };

  // Helper: Find inter-cluster edges
  const findInterClusterEdges = () => {
    const interClusterEdges = [];
    const clusterNodeMap = {};
    
    // Build map of node to cluster
    clusters.forEach(cluster => {
      cluster.nodes.forEach(node => {
        clusterNodeMap[node.id] = cluster.id;
      });
    });
    
    // Find edges between clusters
    const edgeMap = {};
    conceptData.edges.forEach(edge => {
      const sourceCluster = clusterNodeMap[edge.source];
      const targetCluster = clusterNodeMap[edge.target];
      
      if (sourceCluster && targetCluster && sourceCluster !== targetCluster) {
        const key = `${Math.min(sourceCluster, targetCluster)}_${Math.max(sourceCluster, targetCluster)}`;
        if (!edgeMap[key]) {
          edgeMap[key] = {
            sourceCluster: sourceCluster,
            targetCluster: targetCluster,
            count: 0
          };
        }
        edgeMap[key].count++;
      }
    });
    
    return Object.values(edgeMap);
  };

  // Control functions
  const resetView = useCallback(() => {
    if (cyRef.current) {
      cyRef.current.fit(50);
      cyRef.current.center();
    }
  }, []);

  const exportGraph = useCallback(() => {
    if (cyRef.current) {
      const png = cyRef.current.png({
        full: true,
        scale: 2,
        bg: '#ffffff'
      });
      const link = document.createElement('a');
      link.download = `concept-map-${sessionDeviceId}-${new Date().toISOString()}.png`;
      link.href = png;
      link.click();
    }
  }, [sessionDeviceId]);

  if (!sessionDeviceId) {
    return (
      <div className={style.conceptMapContainer}>
        <div className={style.noData}>
          No session device ID provided.
        </div>
      </div>
    );
  }

  return (
    <>
    <div style={{ display: 'flex', height: '100%', width: '100%' }}>
      <div className={style.conceptMapContainer} style={{ flex: 1 }}>
        <div className={style.header}>
          {/* Left: View Toggle + Stats */}
          <div className={style.headerLeft}>
            <div className={style.segmentedControl}>
              <button
                className={viewMode === 'clustered' ? style.segmentedActive : style.segmentedButton}
                onClick={() => setViewMode('clustered')}
              >
                Clustered View
              </button>
              <button
                className={viewMode === 'full' ? style.segmentedActive : style.segmentedButton}
                onClick={() => setViewMode('full')}
              >
                All Concepts
              </button>
            </div>

            <div className={style.statsCompact}>
              <span><strong>{displayData.nodeCount}</strong> concepts</span>
              <span className={style.statsDivider}>|</span>
              <span><strong>{displayData.edgeCount}</strong> connections</span>
            </div>
          </div>

          {/* Right: Controls */}
          <div className={style.headerRight}>
            {viewMode === 'full' && (
              <button
                className={`${style.toggleButton} ${showSpeakerPanel ? style.toggleButtonActive : ''}`}
                onClick={() => setShowSpeakerPanel(!showSpeakerPanel)}
              >
                {showSpeakerPanel ? 'Hide Speakers' : 'Show Speakers'}
              </button>
            )}
            <button
              className={`${style.toggleButton} ${viewLocked ? style.toggleButtonActive : ''}`}
              onClick={() => setViewLocked(!viewLocked)}
              title={viewLocked ? 'Unlock semantic zoom' : 'Lock current view'}
            >
              {viewLocked ? 'View Locked' : 'Lock View'}
            </button>
            <button className={style.actionButton} onClick={resetView}>
              Reset
            </button>
            <button className={style.actionButton} onClick={exportGraph}>
              Export
            </button>
            <button
              className={style.actionButtonPrimary}
              onClick={() => setIsExpanded(!isExpanded)}
            >
              {isExpanded ? 'Exit Fullscreen' : 'Fullscreen'}
            </button>
          </div>
        </div>

        <div
          ref={containerRef}
          className={style.cytoscapeContainer}
          style={{
            height: '600px',
            width: '100%',
            display: isExpanded ? 'none' : 'block'
          }}
        />

        {isLoading && (
          <div className={style.loadingContainer}>
            <div className={style.skeletonGraph}>
              <div className={style.skeletonNode} />
              <div className={style.skeletonNode} />
              <div className={style.skeletonNode} />
              <div className={style.skeletonNode} />
              <div className={style.skeletonNode} />
            </div>
            <div className={style.loadingContent}>
              <div className={style.loadingSpinner}></div>
              <div className={style.loadingText}>Loading Concept Map</div>
            </div>
          </div>
        )}

        {!isLoading && generationStatus === 'processing' && (
          <div className={style.loadingContainer}>
            <div className={style.skeletonGraph}>
              <div className={style.skeletonNode} />
              <div className={style.skeletonNode} />
              <div className={style.skeletonNode} />
              <div className={style.skeletonNode} />
              <div className={style.skeletonNode} />
            </div>
            <div className={style.loadingContent}>
              <div className={style.loadingSpinner}></div>
              <div className={style.loadingText}>Generating Concept Map</div>
              <div className={style.loadingSubtext}>AI is analyzing the full discussion transcript</div>
            </div>
          </div>
        )}

        {!isLoading && generationStatus === 'pending' && conceptData.nodes.length === 0 && (
          <div className={style.noData}>
            <div className={style.loadingText}>Concept map will be generated after the session ends</div>
            <div className={style.loadingSubtext}>The AI analyzes the full discussion for better quality insights</div>
          </div>
        )}

        {!isLoading && generationStatus === 'failed' && (
          <div className={style.noData}>
            <div className={style.loadingText} style={{ color: '#EF4444' }}>Failed to generate concept map</div>
            <div className={style.loadingSubtext}>Please try refreshing the page or contact support</div>
          </div>
        )}

        {!isLoading && generationStatus === 'completed' && conceptData.nodes.length === 0 && (
          <div className={style.noData}>
            No concepts were extracted from the discussion.
          </div>
        )}
      </div>
      
      {showSpeakerPanel && viewMode === 'full' && (
        <SpeakerPanel 
          nodes={conceptData.nodes}
          onSpeakerSelect={setSelectedSpeakers}
          selectedSpeakers={selectedSpeakers}
        />
      )}
      
    </div>

      {isExpanded && (
        <div
          className={style.expandedOverlay}
          onClick={(e) => {
            if (e.target === e.currentTarget) {
              setIsExpanded(false);
            }
          }}
        >
          <div className={style.expandedContainer}>
            <div className={style.expandedHeader}>
              <div className={style.info}>
                <span>Concepts: <strong>{displayData.nodeCount}</strong></span>
                <span>Edges: <strong>{displayData.edgeCount}</strong></span>
              </div>
              <div className={style.controls}>
                <button className={style.actionButton} onClick={resetView}>
                  Reset View
                </button>
                <button className={style.actionButtonPrimary} onClick={() => setIsExpanded(false)}>
                  Exit Fullscreen
                </button>
              </div>
            </div>
            <div
              ref={expandedContainerRef}
              className={style.expandedCytoscapeContainer}
              style={{
                height: 'calc(100% - 60px)',
                width: '100%'
              }}
            />
          </div>
              
    </div>

)}   
{/* Hover Tooltip */}
      {tooltip.visible && tooltip.data && (
        <div
          className={`${style.nodeTooltip} ${tooltip.visible ? style.visible : ''}`}
          style={{ left: tooltip.x, top: tooltip.y }}
        >
          <div className={style.tooltipTitle}>{tooltip.data.label}</div>
          <div className={style.tooltipMeta}>
            <span
              className={style.tooltipType}
              style={{ backgroundColor: `${tooltip.data.color}20`, color: tooltip.data.color }}
            >
              {tooltip.data.type}
            </span>
            {tooltip.data.speakerAlias && (
              <span style={{ color: '#64748B' }}>by {tooltip.data.speakerAlias}</span>
            )}
            {tooltip.data.timestamp > 0 && (
              <span>{Math.floor(tooltip.data.timestamp)}s</span>
            )}
          </div>
        </div>
      )}

{transcriptPanel && (
        <div className={style.transcriptPanel}>
          <div className={style.transcriptHeader}>
            <h3>Source Transcripts</h3>
            <div className={style.transcriptConcept}>
              {transcriptPanel.nodeText}
            </div>
            <div className={style.transcriptTime}>
              At {Math.floor(transcriptPanel.timestamp)}s into discussion
            </div>
            <button
              className={style.transcriptCloseBtn}
              onClick={() => {
                setTranscriptPanel(null);
                setPanelTranscripts([]);
              }}
            >
              Close
            </button>
          </div>
          <div className={style.transcriptList}>
            {transcriptsLoading ? (
              <div className={style.loading}>
                <div className={style.loadingSpinner}></div>
                <div className={style.loadingText}>Loading transcripts...</div>
              </div>
            ) : transcriptsError ? (
              <div className={style.noData}>
                <div className={style.loadingText}>{transcriptsError}</div>
              </div>
            ) : panelTranscripts.length > 0 ? (
              panelTranscripts.map((transcript, idx) => {
                const isHighlighted = Math.abs(transcript.start_time - transcriptPanel.timestamp) <= 5;
                return (
                  <div
                    key={idx}
                    className={`${style.transcriptItem} ${isHighlighted ? style.highlighted : ''}`}
                  >
                    <div className={style.transcriptItemMeta}>
                      <span className={style.speakerDot}></span>
                      <span>{transcript.speaker_alias || 'Unknown'}</span>
                      <span>{Math.floor(transcript.start_time)}s</span>
                    </div>
                    <div className={style.transcriptItemText}>
                      {transcript.transcript}
                    </div>
                  </div>
                );
              })
            ) : (
              <div className={style.noData}>
                <div className={style.loadingText}>No transcripts found in this time range.</div>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}

export default ConceptMapView;
export { ConceptMapView };