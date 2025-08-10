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
  let isProcessing = false; // Indicates if any server operation (upload, query, analysis) is active
  let customPrompts = {};

  // --- NEW: Analysis Polling Variables ---
  let analysisPollingTimer = null;
  const ANALYSIS_POLLING_INTERVAL_MS = 5000; // Poll every 5 seconds for analysis status
  let analysisResultFetched = false;

  function addMessage(content, type = "status") {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message");
    const contentDiv = document.createElement("div");
    contentDiv.classList.add("content");
    // Only set textContent if type is status, otherwise append contentDiv later
    if (type === "status") {
      messageDiv.textContent = content;
    } else {
      contentDiv.innerHTML = content.replace(/\n/g, "<br>"); // Preserve newlines for bot messages
    }

    if (type === "user") {
      messageDiv.classList.add("user-message");
    } else if (type === "bot") {
      messageDiv.classList.add("bot-message");
    } else if (type === "analysis-json") {
        messageDiv.classList.add("analysis-json-message");
        contentDiv.innerHTML = content; // Assuming content is pre-formatted HTML/JSON string
    } else {
        // For status messages, content is already directly in messageDiv.textContent
    }

    if (type !== "status") { // Append contentDiv only if it's user, bot, or analysis-json
        messageDiv.appendChild(contentDiv);
    }
    chatbox.appendChild(messageDiv);
    chatbox.scrollTop = chatbox.scrollHeight;
  }

  // Modified setProcessingState to better reflect analysis state
  function setProcessingState(processing, isAnalysisRunning = false) {
    isProcessing = processing;
    const sessionActive = !!currentSessionId;

    // Query inputs are disabled if any processing is happening OR if analysis is specifically running
    // The query part is only enabled when processing is false AND session is active AND analysis is NOT running
    const canQuery = !processing && sessionActive && !isAnalysisRunning;
    
    queryInput.disabled = !canQuery;
    sendButton.disabled = !canQuery;
    analysisFocusSelect.disabled = !canQuery; // User can't change focus while analysis is pending/running

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
    if (analysisPollingTimer) {
      clearInterval(analysisPollingTimer);
      analysisPollingTimer = null;
    }
    analysisResultFetched = false; // NEW: Reset this flag
    setProcessingState(false);
  }

  // --- MODIFIED: Function to display the full JSON analysis result ---
  async function displayAnalysisResult(sessionId) {
    if (analysisResultFetched) { // Safeguard: if already fetched, exit
        console.log("Analysis result already fetched, skipping duplicate display.");
        return;
    }
    analysisResultFetched = true; // Mark as fetched

    addMessage("Fetching detailed analysis result...", "status");
    try {
        const response = await fetch(`/get_analysis_result/${sessionId}`);
        const result = await response.json();

        if (response.ok && result.analysis_status === "completed") {
            addMessage("Detailed analysis complete! Displaying results:", "status");
            const formattedJson = syntaxHighlight(JSON.stringify(result.analysis_data, null, 2));
            addMessage(`<pre>${formattedJson}</pre>`, "analysis-json");
            
            setProcessingState(false, false);
            if (currentSessionId) queryInput.focus();

        } else {
            analysisResultFetched = false; // Allow re-attempt if result fetch failed
            addMessage(`Failed to retrieve final analysis result: ${result.message || "Unknown error"}`, "status");
            setProcessingState(false, true);
        }
    } catch (error) {
        console.error("Error fetching analysis result:", error);
        analysisResultFetched = false; // Allow re-attempt if network error
        addMessage("Network or server error while fetching analysis result.", "status");
        setProcessingState(false, true);
    }
  }

  // --- MODIFIED: Polling function for analysis status ---
  async function pollAnalysisStatus(sessionId) {
    // If a result has been successfully fetched, stop polling and don't re-enter.
    if (analysisResultFetched) {
        clearInterval(analysisPollingTimer);
        analysisPollingTimer = null;
        return; 
    }

    try {
        const response = await fetch(`/get_analysis_status/${sessionId}`);
        const result = await response.json();

        if (!response.ok) {
            console.error("Polling error:", result);
            addMessage(`Analysis status check failed: ${result.message || "Server error"}`, "status");
            clearInterval(analysisPollingTimer);
            analysisPollingTimer = null;
            setProcessingState(false, true);
            return;
        }

        uploadStatus.textContent = `Analysis Status: ${result.analysis_status.replace(/_/g, ' ')}`;
        uploadStatus.style.color = "#666";

        if (result.analysis_status === "completed") {
            clearInterval(analysisPollingTimer); // CRITICAL: Stop polling FIRST
            analysisPollingTimer = null; // Set to null immediately
            addMessage("Analysis completed successfully on server. Preparing to display results...", "status");
            uploadStatus.textContent = `✅ Analysis Complete: "${result.message}"`;
            uploadStatus.style.color = "green";
            // Call displayAnalysisResult. It will handle the final processing state update.
            displayAnalysisResult(sessionId);
        } else if (result.analysis_status === "failed") {
            clearInterval(analysisPollingTimer);
            analysisPollingTimer = null;
            addMessage(`Detailed analysis failed: ${result.analysis_error || result.message || "Unknown error"}`, "status");
            uploadStatus.textContent = `❌ Analysis Failed: ${result.analysis_error || result.message || "Unknown error"}`;
            uploadStatus.style.color = "red";
            setProcessingState(false, true);
        } else {
            addMessage(`Analysis in progress (${result.analysis_status})... Please wait.`, "status");
            setProcessingState(true, true);
        }

    } catch (error) {
        console.error("Network error during analysis status polling:", error);
        addMessage("Network error during analysis status check. Please check your connection.", "status");
        clearInterval(analysisPollingTimer);
        analysisPollingTimer = null;
        setProcessingState(false, true);
    }
  }

  // --- NEW: Function to syntax highlight JSON for better readability ---
  function syntaxHighlight(json) {
    if (typeof json != 'string') {
        json = JSON.stringify(json, undefined, 2);
    }
    json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
        let cls = 'number';
        if (/^"/.test(match)) {
            if (/:$/.test(match)) {
                cls = 'key';
            } else {
                cls = 'string';
            }
        } else if (/true|false/.test(match)) {
            cls = 'boolean';
        } else if (/null/.test(match)) {
            cls = 'null';
        }
        return '<span class="' + cls + '">' + match + '</span>';
    });
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
    // Insert new custom option before the "---" separator
    const addCustomOption = analysisFocusSelect.querySelector('option[value="add_custom"]');
    if (addCustomOption) {
        analysisFocusSelect.insertBefore(newOption, addCustomOption);
    } else {
        // Fallback if separator is not found
        analysisFocusSelect.appendChild(newOption);
    }

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
      chatbox.innerHTML = ""; // Clear chat for a fresh session
    }
    addMessage(`Uploading "${file.name}"...`, "status");
    uploadStatus.textContent = `Uploading "${file.name}"...`;
    uploadStatus.style.color = "#666";
    cleanupStatus.textContent = "";
    setProcessingState(true, true); // Indicate processing and analysis running
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch("/upload", { method: "POST", body: formData });
      const result = await response.json();
      if (response.ok && result.success) {
        currentSessionId = result.session_id;
        // The upload is successful, but analysis is still in background
        addMessage(`Document "${result.filename}" uploaded. Starting detailed analysis...`, "status");
        uploadStatus.textContent = `Uploading complete. Analysis status: ${result.analysis_status.replace(/_/g, ' ')}`;
        uploadStatus.style.color = "#666";

        // Start polling for analysis status
        analysisPollingTimer = setInterval(() => pollAnalysisStatus(currentSessionId), ANALYSIS_POLLING_INTERVAL_MS);
        pollAnalysisStatus(currentSessionId); // Initial immediate poll

        // Keep query input disabled until analysis is complete
        setProcessingState(true, true); // Still processing, analysis is running
      } else {
        uploadStatus.textContent = `❌ Error: ${result.message || "Upload failed"}`;
        uploadStatus.style.color = "red";
        addMessage(`Error processing document: ${result.message}.`, "status");
        setProcessingState(false, true); // Failed, so disable query inputs
      }
    } catch (error) {
      console.error("Upload error:", error);
      uploadStatus.textContent = "❌ Network or server error during upload.";
      uploadStatus.style.color = "red";
      addMessage("Network or server error during upload. Check console.", "status");
      setProcessingState(false, true); // Disable query inputs
    }
  });

  async function submitQuery() {
    const query = queryInput.value.trim();
    const focusAreaValue = analysisFocusSelect.value;
    // Ensure analysis is complete before allowing queries
    if (!query || !currentSessionId || isProcessing || !focusAreaValue || analysisPollingTimer) {
      // If analysisPollingTimer is active, analysis is still running
      addMessage("Please wait for the detailed analysis to complete before asking questions.", "status");
      return;
    }
    
    const selectedFocusText =
      analysisFocusSelect.options[analysisFocusSelect.selectedIndex].text;
    addMessage(`${query} (Focus: ${selectedFocusText})`, "user");
    queryInput.value = "";
    setProcessingState(true, false); // Only general processing, analysis is NOT running anymore
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
      console.log("QUERY RESULT (PARSED JSON):", result);

      addMessage(result.answer, "bot");
      
      // Display OpenAI sources if available
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
      
      // Local sources are now typically excluded from results, but keeping the log for robustness
      if (result.local_sources && result.local_sources.length > 0) {
        console.log("Local COEQWAL sources retrieved (not displayed):", result.local_sources);
      }

    } catch (error) {
      console.error("QUERY CATCH BLOCK: An error occurred:", error);
      addMessage(`An error occurred: ${error.message}`, "status");
    } finally {
      console.log("QUERY FINALLY BLOCK: Re-enabling controls.");
      setProcessingState(false, false); // No general processing, no analysis running
      if (currentSessionId) queryInput.focus();
    }
  }

  sendButton.addEventListener("click", submitQuery);
  queryInput.addEventListener("keypress", (event) => { if (event.key === "Enter") submitQuery(); });

  endChatButton.addEventListener("click", async () => {
    if (!currentSessionId || isProcessing) return;
    addMessage("Ending session...", "status");
    setProcessingState(true, true); // Indicate processing, analysis might still be running or ending
    cleanupStatus.textContent = "Cleaning up...";
    cleanupStatus.style.color = "#666";
    
    // Clear any active polling timer
    if (analysisPollingTimer) {
      clearInterval(analysisPollingTimer);
      analysisPollingTimer = null;
    }

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
        setProcessingState(false, false); // Reset, but query might be disabled if backend state is ambiguous
      }
    } catch (error) {
      console.error("End session error:", error);
      cleanupStatus.textContent = "❌ Network or server error.";
      addMessage("Network or server error. Check console.", "status");
      setProcessingState(false, false); // Reset, but query might be disabled
    }
  });

  resetUI();
});