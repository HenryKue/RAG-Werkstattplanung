import { useState } from 'react'

import '@chatscope/chat-ui-kit-styles/dist/default/styles.min.css';
import { MainContainer, ChatContainer, MessageList, Message, MessageInput, TypingIndicator } from "@chatscope/chat-ui-kit-react";
import './App.css'

function App() {
  const [typing, setTyping] = useState(false);
  const [messages, setMessages] = useState([
    {
      message: "Hallo, ich bin ein digitaler Reparaturassistent. Wie kann ich dir helfen?",
      sender: "AI",
      direction: "incoming"
    }
  ])

  const handleSend = async (message) => {
    const newMessage = {
      message: message,
      sender: "user",
      direction: "outgoing"
    }
    const newMessages = [...messages, newMessage];
    setMessages(newMessages);
    setTyping(true);
    processMessageToAI(newMessage);
  }

  async function processMessageToAI(message) {
    const apiRequestBody = {
      "query": message.message
    }

    await fetch("http://192.168.178.33:8000/query", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(apiRequestBody)
    }).then((data) => {
      return data.json();
    }).then((data) => {


      const answerText = data.Answer;
      const imagePaths = data.Images || [];

      const aiMessage = {
        message: answerText,
        sender: "AI",
        direction: "incoming",
        images: imagePaths
      }


      setMessages(prev => [...prev, aiMessage]);
      setTyping(false); 
    });
  }

  return (
    <div className='App'>
      <div className='chat-wrapper'>
        <MainContainer>
          <ChatContainer>
            <MessageList
              scrollBehavior='smooth'
              typingIndicator={typing ? <TypingIndicator content="KI denkt nach" /> : null}
            >
              {messages.map((message, i) => {
                const hasImages = message.images && message.images.length > 0;
                            
                if (message.sender === "AI" && hasImages) {
                  return (
                    <div className="ai-message" key={i}>
                      <Message model={message} />
                      <div className="image-grid">
                        {message.images.map((imgPath, j) => (
                          <img
                            key={j}
                            src={`/${imgPath}`}
                            alt={`Bild ${j + 1}`}
                            onError={(e) => (e.target.style.display = "none")}
                          />
                        ))}
                      </div>
                    </div>
                  );
                }
              
                return <Message key={i} model={message} />;
              })}
            </MessageList>
            <MessageInput
              placeholder='Gib ein Stichwort oder eine konkrete Frage ein...'
              onSend={handleSend}
              attachButton={false}
            />
          </ChatContainer>
        </MainContainer>
      </div>
    </div>
  );
}

export default App
