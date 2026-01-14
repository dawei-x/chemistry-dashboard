/**
 * Message Bubble Component
 *
 * Displays a single message in the chat interface.
 * Enhanced with clickable inline citations.
 */

import React, { useState } from 'react';
import ReasoningTrace from './ReasoningTrace';
import ReferenceList from './ReferenceList';
import styles from './MessageBubble.module.css';

const MessageBubble = ({ message, sessionDeviceId, onCitationClick, onReferenceClick }) => {
    const [showReasoning, setShowReasoning] = useState(false);
    const [showReferences, setShowReferences] = useState(false);

    const isUser = message.role === 'user' || message.isUser;
    const hasReasoning = message.reasoning_trace && message.reasoning_trace.length > 0;
    const hasCitations = message.citations && message.citations.length > 0;
    const hasTools = message.tools_used && message.tools_used.length > 0;
    const hasReferences = message.references && message.references.length > 0;

    // Format confidence as percentage
    const confidencePercent = message.confidence
        ? Math.round(message.confidence * 100)
        : null;

    return (
        <div className={`${styles.bubble} ${isUser ? styles.userBubble : styles.assistantBubble}`}>
            {/* Message content - citations shown in References section below */}
            <div className={styles.content}>
                {renderContent(message.content)}
            </div>

            {/* Assistant message extras */}
            {!isUser && (
                <>
                    {/* V4 References (tool results as transparency) */}
                    {hasReferences && (
                        <div className={styles.referencesSection}>
                            <button
                                className={styles.referencesToggle}
                                onClick={() => setShowReferences(!showReferences)}
                            >
                                {showReferences ? 'Hide sources' : `View sources (${message.references.length})`}
                            </button>
                            {showReferences && (
                                <div className={styles.referencesList}>
                                    {message.references.map((ref, idx) => (
                                        <ReferenceItem
                                            key={idx}
                                            reference={ref}
                                            onClick={onReferenceClick}
                                        />
                                    ))}
                                </div>
                            )}
                        </div>
                    )}

                    {/* Legacy Citation List for backwards compatibility */}
                    {hasCitations && (
                        <ReferenceList
                            citations={message.citations}
                            onCitationClick={onCitationClick}
                        />
                    )}

                    {/* Metadata bar */}
                    <div className={styles.metadata}>
                        {confidencePercent !== null && (
                            <span
                                className={`${styles.confidence} ${
                                    confidencePercent >= 80 ? styles.high :
                                    confidencePercent >= 50 ? styles.medium : styles.low
                                }`}
                                title={`Confidence: ${confidencePercent}%`}
                            >
                                {confidencePercent}% confident
                            </span>
                        )}

                        {hasTools && (
                            <span className={styles.tools} title={message.tools_used.join(', ')}>
                                {message.tools_used.length} tool{message.tools_used.length !== 1 ? 's' : ''} used
                            </span>
                        )}

                        {hasReasoning && (
                            <button
                                className={styles.reasoningToggle}
                                onClick={() => setShowReasoning(!showReasoning)}
                            >
                                {showReasoning ? 'Hide reasoning' : 'Show reasoning'}
                            </button>
                        )}
                    </div>

                    {/* Reasoning trace */}
                    {showReasoning && hasReasoning && (
                        <ReasoningTrace trace={message.reasoning_trace} />
                    )}
                </>
            )}

            {/* Timestamp */}
            {message.created_at && (
                <div className={styles.timestamp}>
                    {formatTime(message.created_at)}
                </div>
            )}
        </div>
    );
};

/**
 * Render message content with basic markdown support.
 */
const renderContent = (content) => {
    if (!content) return null;

    // Split into paragraphs
    const paragraphs = content.split(/\n\n+/);

    return paragraphs.map((para, idx) => {
        // Check for bullet points
        if (para.includes('\n- ') || para.startsWith('- ')) {
            const items = para.split('\n').filter(line => line.startsWith('- '));
            return (
                <ul key={idx} className={styles.list}>
                    {items.map((item, i) => (
                        <li key={i}>{formatInline(item.substring(2))}</li>
                    ))}
                </ul>
            );
        }

        // Check for numbered list
        if (/^\d+\.\s/.test(para)) {
            const items = para.split('\n').filter(line => /^\d+\.\s/.test(line));
            return (
                <ol key={idx} className={styles.list}>
                    {items.map((item, i) => (
                        <li key={i}>{formatInline(item.replace(/^\d+\.\s*/, ''))}</li>
                    ))}
                </ol>
            );
        }

        // Regular paragraph
        return <p key={idx} className={styles.paragraph}>{formatInline(para)}</p>;
    });
};

/**
 * Format inline elements (bold, italic).
 * Note: Inline citations are no longer rendered - citations appear in References section below.
 */
const formatInline = (text) => {
    if (!text) return null;

    // Handle bold text
    const boldParts = text.split(/\*\*(.*?)\*\*/g);
    if (boldParts.length > 1) {
        return boldParts.map((part, i) => {
            if (i % 2 === 1) {
                return <strong key={i}>{part}</strong>;
            }
            return part;
        });
    }

    return text;
};

/**
 * Format timestamp for display.
 */
const formatTime = (isoString) => {
    const date = new Date(isoString);
    const now = new Date();
    const isToday = date.toDateString() === now.toDateString();

    if (isToday) {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    return date.toLocaleDateString([], { month: 'short', day: 'numeric' }) +
           ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

/**
 * Reference Item for V4 tool results.
 * Displays source data the agent used.
 */
const ReferenceItem = ({ reference, onClick }) => {
    const handleClick = () => {
        if (onClick && reference.clickable) {
            onClick(reference);
        }
    };

    // Icons for different reference types
    const getIcon = (type) => {
        switch (type) {
            case 'transcript': return '📝';
            case 'concept_map': return '🗺️';
            case 'collaboration': return '📊';
            case 'speaker_utterances': return '🗣️';
            case 'speaker_profile': return '👤';
            case 'comparison': return '⚖️';
            case 'search': return '🔍';
            case 'session_list': return '📋';
            default: return '📎';
        }
    };

    return (
        <div
            className={`${styles.referenceItem} ${reference.clickable ? styles.clickable : ''}`}
            onClick={handleClick}
            role={reference.clickable ? 'button' : undefined}
            tabIndex={reference.clickable ? 0 : undefined}
        >
            <span className={styles.refIcon}>{getIcon(reference.type)}</span>
            <span className={styles.refSummary}>{reference.summary}</span>
            {reference.session_name && (
                <span className={styles.refSession}>{reference.session_name}</span>
            )}
        </div>
    );
};

export default MessageBubble;
