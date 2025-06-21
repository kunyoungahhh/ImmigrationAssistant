"use client"

import type React from "react"
import { memo } from "react";
import { useState, useRef, useEffect } from "react"
import "./App.css"
import ChatMessage from "./components/ChatMessage.tsx"
import ChatOptions from './components/ChatOptions.tsx'
import ExampleQuestions from "./components/ExampleQuestions.tsx"
import type { Message } from "./types"
import Iridescence from "./components/Iridescence.tsx";
import {useMemo} from "react";

function App() {

  const MemoIridescence = memo(Iridescence);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      role: "assistant",
      content:
        "Hello! I'm your immigration law assistant. Ask me any question about immigration law, and I'll try to answer based on the documents I've been trained on.",
      sources: [],
      timestamp: new Date(),
    },
  ])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const backgroundColor = useMemo(() => [0.05, 0.15, 0.35], []);
  const [selectedOption, setSelectedOption] = useState<string | null>(null);

  // Scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim()) return

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
      sources: [],
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsLoading(true)

    try {
      const response = await fetch("http://localhost:5001/api/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: input, topic: selectedOption }),
      })

      if (!response.ok) {
        throw new Error("Failed to get response")
      }

      const data = await response.json()

      // Add assistant message
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.answer,
        sources: data.sources || [],
        processingTime: data.processing_time,
        timestamp: new Date(),
      }

      setMessages((prev) => [...prev, assistantMessage])
    } catch (error) {
      console.error("Error:", error)

      // Add error message
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "Sorry, there was an error processing your request. Please try again.",
        sources: [],
        timestamp: new Date(),
        isError: true,
      }

      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleExampleClick = (question: string) => {
    setInput(question)
  }

  return (
            <div className="main-container"><Iridescence color={backgroundColor} speed={0.5} amplitude={0.1} />
    <div className="relative app-container min-h-screen overflow-hidden z-10">
      <header className="app-header">
        <h1>Immigration Law Assistant</h1>
        <p>Ask questions about immigration law and get answers based on legal documents</p>
      </header>

      <main className="chat-container">
        <div id="chat-options" className="mb-6">
          <ChatOptions onOptionSelect={(option) => setSelectedOption(option)} />
        </div>
        <div className="messages-container">
          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))}

          {isLoading && (
            <ChatMessage
              message={{
                id: "loading",
                role: "assistant",
                content: "Thinking...",
                sources: [],
                timestamp: new Date(),
                isLoading: true,
              }}
            />
          )}

          <div ref={messagesEndRef} />
        </div>

        <form onSubmit={handleSubmit} className="input-form">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about immigration law..."
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading || !input.trim()}>
            {isLoading ? "Thinking..." : "Ask"}
          </button>
        </form>

        <ExampleQuestions onExampleClick={handleExampleClick} />
      </main>

      <footer className="app-footer">
        <p>
          This assistant provides information based on specific legal documents. It is not a substitute for professional
          legal advice.
        </p>
      </footer>
      </div>
              </div>
  )
}

export default App
