/**
 * Message Bubble Component
 *
 * Displays a single message in the chat interface.
 */

import React, { useState } from 'react';
import CitationCard from './CitationCard';
import ReasoningTrace from './ReasoningTrace';
import styles from './MessageBubble.module.css';

const MessageBubble = ({ message, sessionDeviceId }) => {
    const [showReasoning, setShowReasoning] = useState(false);

    const isUser = message.role === 'user' || message.isUser;
    const hasReasoning = message.reasoning_trace && message.reasoning_trace.length > 0;
    const hasCitations = message.citations && message.citations.length > 0;
    const hasTools = message.tools_used && message.tools_used.length > 0;

    // Format confidence as percentage
    const confidencePercent = message.confidence
        ? Math.round(message.confidence * 100)
        : null;

    return (
        <div className={`${styles.bubble} ${isUser ? styles.userBubble : styles.assistantBubble}`}>
            {/* Message content */}
            <div className={styles.content}>
                {renderContent(message.content)}
            </div>

            {/* Assistant message extras */}
            {!isUser && (
                <>
                    {/* Citations */}
                    {hasCitations && (
                        <div className={styles.citations}>
                            {message.citations.map((citation, idx) => (
                                <CitationCard
                                    key={idx}
                                    citation={citation}
                                    sessionDeviceId={sessionDeviceId}
                                />
                            ))}
                        </div>
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
 * Format inline elements (bold, italic, citations).
 */
const formatInline = (text) => {
    if (!text) return null;

    // Handle bold
    let parts = text.split(/\*\*(.*?)\*\*/g);
    if (parts.length > 1) {
        return parts.map((part, i) => (
            i % 2 === 1 ? <strong key={i}>{part}</strong> : part
        ));
    }

    // Handle inline citations [Session X]
    parts = text.split(/\[([^\]]+)\]/g);
    if (parts.length > 1) {
        return parts.map((part, i) => (
            i % 2 === 1 ? <span key={i} className={styles.inlineCitation}>[{part}]</span> : part
        ));
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

export default MessageBubble;
