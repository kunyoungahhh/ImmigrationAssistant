import type React from "react"
import type { Message } from "../types"
import "./ChatMessage.css"

interface ChatMessageProps {
  message: Message
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const { role, content, sources, processingTime, isLoading, isError } = message

  const avatarText = role === "user" ? "You" : "AI"
  const avatarClass = role === "user" ? "user-avatar" : "assistant-avatar"
  const messageClass = role === "user" ? "user-message" : "assistant-message"

  return (
    <div className={`message-container ${role}`}>
      <div className={`avatar ${avatarClass}`}>{avatarText}</div>
      <div className={`message ${messageClass} ${isError ? "error" : ""}`}>
        {isLoading ? (
          <div className="loading-indicator">
            <span className="dot"></span>
            <span className="dot"></span>
            <span className="dot"></span>
          </div>
        ) : (
          <>
            <div className="message-content">{content}</div>

            {sources && sources.length > 0 && (
              <div className="message-sources">
                <span className="sources-label">Sources:</span> {sources.join(", ")}
              </div>
            )}

            {processingTime && <div className="processing-time">Processed in {processingTime}</div>}
          </>
        )}
      </div>
    </div>
  )
}

export default ChatMessage
