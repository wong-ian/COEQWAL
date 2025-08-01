// static/app.js
document.addEventListener("DOMContentLoaded", () => {
  const uploadButton = document.getElementById("upload-button");
  const fileInput = document.getElementById("document-upload");
  const uploadStatus = document.getElementById("upload-status");
  const chatbox = document.getElementById("chatbox");
  const queryInput = document.getElementById("query-input");
  const sendButton = document.getElementById("send-button");
  const thinkingIndicator = document.getElementById("thinking-indicator");
  const endChatButton = document.getElementById("end-chat-button");
  const cleanupStatus = document.getElementById("cleanup-status");
  const analysisFocusSelect = document.getElementById("analysis-focus");
  const customFocusModal = document.getElementById("custom-focus-modal");
  const customFocusNameInput = document.getElementById("custom-focus-name");
  const customFocusInstructionsInput = document.getElementById(
    "custom-focus-instructions",
  );
  const modalSaveButton = document.getElementById("modal-save-button");
  const modalCancelButton = document.getElementById("modal-cancel-button");

  let currentSessionId = null;
  let isProcessing = false;
  let customPrompts = {};

  function addMessage(content, type = "status") {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message");
    const contentDiv = document.createElement("div");
    contentDiv.classList.add("content");
    contentDiv.textContent = content;
    if (type === "user") {
      messageDiv.classList.add("user-message");
    } else if (type === "bot") {
      messageDiv.classList.add("bot-message");
      contentDiv.innerHTML = contentDiv.innerHTML.replace(/\n/g, "<br>");
    } else {
      messageDiv.classList.add("status-message");
      messageDiv.textContent = content;
    }
    if (type !== "status") {
      messageDiv.appendChild(contentDiv);
    }
    chatbox.appendChild(messageDiv);
    chatbox.scrollTop = chatbox.scrollHeight;
  }

  function setProcessingState(processing) {
    isProcessing = processing;
    const sessionActive = !!currentSessionId;
    const canInteract = !processing && sessionActive;
    queryInput.disabled = !canInteract;
    sendButton.disabled = !canInteract;
    analysisFocusSelect.disabled = !canInteract;
    uploadButton.disabled = processing;
    fileInput.disabled = processing;
    endChatButton.disabled = processing || !sessionActive;
    thinkingIndicator.style.display = processing ? "block" : "none";
  }

  function resetUI() {
    currentSessionId = null;
    customPrompts = {};
    const customOptions = analysisFocusSelect.querySelectorAll(
      'option[value^="custom_"]',
    );
    customOptions.forEach((option) => option.remove());
    chatbox.innerHTML =
      '<div class="status-message">Please upload a document and select an analysis focus to begin.</div>';
    uploadStatus.textContent = "";
    cleanupStatus.textContent = "";
    fileInput.value = "";
    analysisFocusSelect.value = "general";
    setProcessingState(false);
  }

  analysisFocusSelect.addEventListener("change", () => {
    if (analysisFocusSelect.value === "add_custom") {
      customFocusModal.style.display = "flex";
      analysisFocusSelect.value = "general";
    }
  });

  modalCancelButton.addEventListener("click", () => {
    customFocusModal.style.display = "none";
  });

  modalSaveButton.addEventListener("click", () => {
    const name = customFocusNameInput.value.trim();
    const instructions = customFocusInstructionsInput.value.trim();
    if (!name || !instructions) {
      alert("Please provide both a name and instructions.");
      return;
    }
    const customId = `custom_${Date.now()}`;
    customPrompts[customId] = instructions;
    const newOption = document.createElement("option");
    newOption.value = customId;
    newOption.textContent = name;
    analysisFocusSelect.insertBefore(
      newOption,
      analysisFocusSelect.options[analysisFocusSelect.options.length - 2],
    );
    analysisFocusSelect.value = customId;
    customFocusNameInput.value = "";
    customFocusInstructionsInput.value = "";
    customFocusModal.style.display = "none";
  });

  uploadButton.addEventListener("click", async () => {
    const file = fileInput.files[0];
    if (!file) {
      uploadStatus.textContent = "Please select a file first.";
      uploadStatus.style.color = "red";
      return;
    }
    if (currentSessionId) {
      addMessage("Uploading new document...", "status");
    } else {
      chatbox.innerHTML = "";
    }
    addMessage(`Uploading "${file.name}"...`, "status");
    uploadStatus.textContent = `Uploading "${file.name}"...`;
    uploadStatus.style.color = "#666";
    cleanupStatus.textContent = "";
    setProcessingState(true);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch("/upload", { method: "POST", body: formData });
      const result = await response.json();
      if (response.ok && result.success) {
        currentSessionId = result.session_id;
        uploadStatus.textContent = `✅ Ready: "${result.filename}" (Session Active)`;
        uploadStatus.style.color = "green";
        if (chatbox.innerHTML.includes("Please upload a document")) {
          chatbox.innerHTML = "";
        }
        addMessage(`Document processed. Select focus and ask questions.`, "status");
        setProcessingState(false);
        queryInput.focus();
      } else {
        uploadStatus.textContent = `❌ Error: ${result.message || "Upload failed"}`;
        uploadStatus.style.color = "red";
        addMessage(`Error processing document: ${result.message}.`, "status");
        setProcessingState(false);
        queryInput.disabled = true;
        sendButton.disabled = true;
        analysisFocusSelect.disabled = true;
      }
    } catch (error) {
      console.error("Upload error:", error);
      uploadStatus.textContent = "❌ Network or server error.";
      uploadStatus.style.color = "red";
      addMessage("Network or server error. Check console.", "status");
      setProcessingState(false);
      queryInput.disabled = true;
      sendButton.disabled = true;
      analysisFocusSelect.disabled = true;
    }
  });

  async function submitQuery() {
    const query = queryInput.value.trim();
    const focusAreaValue = analysisFocusSelect.value;
    if (!query || !currentSessionId || isProcessing || !focusAreaValue) {
      return;
    }
    const selectedFocusText =
      analysisFocusSelect.options[analysisFocusSelect.selectedIndex].text;
    addMessage(`${query} (Focus: ${selectedFocusText})`, "user");
    queryInput.value = "";
    setProcessingState(true);
    cleanupStatus.textContent = "";
    let payload = {
      session_id: currentSessionId,
      query: query,
      focus_area: focusAreaValue,
      custom_instructions: null,
    };
    if (focusAreaValue.startsWith("custom_")) {
      payload.focus_area = "custom";
      payload.custom_instructions = customPrompts[focusAreaValue];
    }

    try {
      console.log("SUBMITTING QUERY: Sending payload to /query:", payload);
      const response = await fetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      console.log("QUERY RESPONSE: Received response from server with status:", response.status);

      if (!response.ok) {
        let errorMsg = `Server responded with an error: ${response.status}`;
        try { const errData = await response.json(); errorMsg = errData.detail || errorMsg; } catch (e) {}
        throw new Error(errorMsg);
      }

      const result = await response.json();
      console.log("QUERY RESULT (PARSED JSON):", result); // Log the entire received object

      addMessage(result.answer, "bot");
      
      // --- ADDED THIS MORE DETAILED LOGGING BLOCK ---
      console.log("Checking for 'openai_sources' in the result...");
      if (result.openai_sources && Array.isArray(result.openai_sources) && result.openai_sources.length > 0) {
        console.log(`SUCCESS: Found ${result.openai_sources.length} OpenAI sources. Creating display element...`);
        const sourcesContainer = document.createElement("div");
        sourcesContainer.classList.add("bot-sources");
        const title = document.createElement("strong");
        title.textContent = "Sources from your document:";
        sourcesContainer.appendChild(title);

        result.openai_sources.forEach((sourceHTML) => {
          console.log("  - Adding source:", sourceHTML);
          const sourceElement = document.createElement("div");
          sourceElement.innerHTML = sourceHTML;
          sourcesContainer.appendChild(sourceElement);
        });

        chatbox.appendChild(sourcesContainer);
        chatbox.scrollTop = chatbox.scrollHeight;
      } else {
        console.warn("WARNING: 'openai_sources' key was not found or the array is empty in the response from the server.", "Received value:", result.openai_sources);
      }
      
      if (result.local_sources && result.local_sources.length > 0) {
        console.log("Local COEQWAL sources retrieved:", result.local_sources);
      }

    } catch (error) {
      console.error("QUERY CATCH BLOCK: An error occurred:", error);
      addMessage(`An error occurred: ${error.message}`, "status");
    } finally {
      console.log("QUERY FINALLY BLOCK: Re-enabling controls.");
      setProcessingState(false);
      if (currentSessionId) queryInput.focus();
    }
  }

  sendButton.addEventListener("click", submitQuery);
  queryInput.addEventListener("keypress", (event) => { if (event.key === "Enter") submitQuery(); });

  endChatButton.addEventListener("click", async () => {
    if (!currentSessionId || isProcessing) return;
    addMessage("Ending session...", "status");
    setProcessingState(true);
    cleanupStatus.textContent = "Cleaning up...";
    cleanupStatus.style.color = "#666";
    try {
      const response = await fetch("/end-session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: currentSessionId }),
      });
      const result = await response.json();
      if (response.ok && result.success) {
        cleanupStatus.textContent = "✅ Resources cleaned up.";
        addMessage("Session ended. It is safe to close the window.", "status");
        resetUI();
      } else {
        cleanupStatus.textContent = `❌ Cleanup failed: ${result.message || "Unknown error"}`;
        addMessage(`Error ending session: ${result.message}.`, "status");
        setProcessingState(false);
      }
    } catch (error) {
      console.error("End session error:", error);
      cleanupStatus.textContent = "❌ Network or server error.";
      addMessage("Network or server error. Check console.", "status");
      setProcessingState(false);
    }
  });

  resetUI();
});