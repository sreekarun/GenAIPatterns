import { useState, useRef, useEffect } from "react";

type Message = { role: "user" | "assistant" | "error"; content: string };

async function sendMessage(message: string): Promise<string> {
  const res = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || String(res.status));
  }
  const data = await res.json();
  return data.reply ?? "(No response)";
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || loading) return;
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setLoading(true);
    setStatus(null);
    try {
      const reply = await sendMessage(text);
      setMessages((prev) => [...prev, { role: "assistant", content: reply }]);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setMessages((prev) => [...prev, { role: "error", content: msg }]);
      setStatus(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <header className="chat-header">
        <h1>Stocks MCP Chat</h1>
        <p className="status">Ask about stock quotes and company info (e.g. “What is the price of AAPL?”)</p>
        {status && <p className="status error">{status}</p>}
      </header>
      <div className="chat-messages">
        {messages.length === 0 && (
          <p className="status">Send a message to start. Make sure the API is running on port 8000.</p>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`message ${m.role}`}>
            {m.content}
          </div>
        ))}
        {loading && <div className="message assistant">…</div>}
        <div ref={bottomRef} />
      </div>
      <form className="chat-form" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about a stock..."
          disabled={loading}
        />
        <button type="submit" disabled={loading}>
          Send
        </button>
      </form>
    </div>
  );
}
