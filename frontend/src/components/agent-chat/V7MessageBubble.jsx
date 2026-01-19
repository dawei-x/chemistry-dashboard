/**
 * V7 Message Bubble Component
 *
 * V7-specific message display. Simplified - no legacy citations/references.
 * Shows: message content + tools used.
 */

import React from 'react';
import styles from './V7MessageBubble.module.css';

const V7MessageBubble = ({ message, onCitationClick }) => {
    const isUser = message.role === 'user' || message.isUser;
    const hasTools = message.tools_used && message.tools_used.length > 0;

    return (
        <div className={`${styles.bubble} ${isUser ? styles.userBubble : styles.assistantBubble}`}>
            {/* Message content */}
            <div className={styles.content}>
                {renderContent(message.content)}
            </div>

            {/* Assistant message extras */}
            {!isUser && hasTools && (
                <div className={styles.metadata}>
                    <span className={styles.tools} title={message.tools_used.join(', ')}>
                        {message.tools_used.length} tool{message.tools_used.length !== 1 ? 's' : ''} used
                    </span>
                </div>
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

    const paragraphs = content.split(/\n\n+/);

    return paragraphs.map((para, idx) => {
        // Headers
        if (para.startsWith('### ')) {
            return <h4 key={idx} className={styles.heading}>{formatInline(para.substring(4))}</h4>;
        }
        if (para.startsWith('## ')) {
            return <h3 key={idx} className={styles.heading}>{formatInline(para.substring(3))}</h3>;
        }
        if (para.startsWith('# ')) {
            return <h2 key={idx} className={styles.heading}>{formatInline(para.substring(2))}</h2>;
        }

        // Bullet points
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

        // Numbered list
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

export default V7MessageBubble;
