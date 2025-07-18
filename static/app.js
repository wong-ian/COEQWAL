document.addEventListener('DOMContentLoaded', () => {
    const uploadButton = document.getElementById('upload-button');
    const fileInput = document.getElementById('document-upload');
    const uploadStatus = document.getElementById('upload-status');
    const chatbox = document.getElementById('chatbox');
    const queryInput = document.getElementById('query-input');
    const sendButton = document.getElementById('send-button');
    const thinkingIndicator = document.getElementById('thinking-indicator');
    const endChatButton = document.getElementById('end-chat-button');
    const cleanupStatus = document.getElementById('cleanup-status');
    const analysisFocusSelect = document.getElementById('analysis-focus');
    const customFocusModal = document.getElementById('custom-focus-modal');
    const customFocusNameInput = document.getElementById('custom-focus-name');
    const customFocusInstructionsInput = document.getElementById('custom-focus-instructions');
    const modalSaveButton = document.getElementById('modal-save-button');
    const modalCancelButton = document.getElementById('modal-cancel-button');

    let currentSessionId = null;
    let isProcessing = false;
    let customPrompts = {};

    function addMessage(content, type = 'status') {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        const contentDiv = document.createElement('div');
        contentDiv.classList.add('content');
        contentDiv.textContent = content;
        if (type === 'user') {
            messageDiv.classList.add('user-message');
        } else if (type === 'bot') {
            messageDiv.classList.add('bot-message');
            contentDiv.innerHTML = contentDiv.innerHTML.replace(/\n/g, '<br>');
        } else {
            messageDiv.classList.add('status-message');
            messageDiv.textContent = content;
        }
        if (type !== 'status') {
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
        thinkingIndicator.style.display = processing ? 'block' : 'none';
    }

    function resetUI() {
        currentSessionId = null;
        customPrompts = {};
        const customOptions = analysisFocusSelect.querySelectorAll('option[value^="custom_"]');
        customOptions.forEach(option => option.remove());
        chatbox.innerHTML = '<div class="status-message">Please upload a document and select an analysis focus to begin.</div>';
        uploadStatus.textContent = '';
        cleanupStatus.textContent = '';
        fileInput.value = '';
        analysisFocusSelect.value = 'general';
        setProcessingState(false);
    }

    analysisFocusSelect.addEventListener('change', () => {
        if (analysisFocusSelect.value === 'add_custom') {
            customFocusModal.style.display = 'flex';
            analysisFocusSelect.value = 'general';
        }
    });

    modalCancelButton.addEventListener('click', () => {
        customFocusModal.style.display = 'none';
    });

    modalSaveButton.addEventListener('click', () => {
        const name = customFocusNameInput.value.trim();
        const instructions = customFocusInstructionsInput.value.trim();
        if (!name || !instructions) {
            alert('Please provide both a name and instructions for your custom focus.');
            return;
        }
        const customId = `custom_${Date.now()}`;
        customPrompts[customId] = instructions;
        const newOption = document.createElement('option');
        newOption.value = customId;
        newOption.textContent = name;
        analysisFocusSelect.insertBefore(newOption, analysisFocusSelect.options[analysisFocusSelect.options.length - 2]);
        analysisFocusSelect.value = customId;
        customFocusNameInput.value = '';
        customFocusInstructionsInput.value = '';
        customFocusModal.style.display = 'none';
    });

    uploadButton.addEventListener('click', async () => {
        const file = fileInput.files[0];
        if (!file) {
            uploadStatus.textContent = 'Please select a file first.';
            uploadStatus.style.color = 'red';
            return;
        }
        if (currentSessionId) {
             addMessage('Uploading new document. Previous session resources will be replaced on success...', 'status');
        } else {
             chatbox.innerHTML = '';
        }
        addMessage(`Uploading "${file.name}"...`, 'status');
        uploadStatus.textContent = `Uploading "${file.name}"...`;
        uploadStatus.style.color = '#666';
        cleanupStatus.textContent = '';
        setProcessingState(true);
        const formData = new FormData();
        formData.append('file', file);
        try {
            const response = await fetch('/upload', { method: 'POST', body: formData });
            const result = await response.json();
            if (response.ok && result.success) {
                currentSessionId = result.session_id;
                uploadStatus.textContent = `✅ Ready: "${result.filename}" (Session Active)`;
                uploadStatus.style.color = 'green';
                if (chatbox.innerHTML.includes('Please upload a document')) {
                    chatbox.innerHTML = '';
                }
                addMessage(`Document "${result.filename}" processed. Select focus and ask questions.`, 'status');
                setProcessingState(false);
                queryInput.focus();
            } else {
                uploadStatus.textContent = `❌ Error: ${result.message || 'Upload failed'}`;
                uploadStatus.style.color = 'red';
                addMessage(`Error processing document: ${result.message || 'Please try again.'}`, 'status');
                setProcessingState(false);
                queryInput.disabled = true;
                sendButton.disabled = true;
                analysisFocusSelect.disabled = true;
            }
        } catch (error) {
            console.error('Upload error:', error);
            uploadStatus.textContent = '❌ Network or server error during upload.';
            uploadStatus.style.color = 'red';
            addMessage('Network or server error during upload. Check console.', 'status');
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
            if (!focusAreaValue && currentSessionId) {
                addMessage('Please select an analysis focus from the dropdown.', 'status');
            }
            return;
        }
        const selectedFocusText = analysisFocusSelect.options[analysisFocusSelect.selectedIndex].text;
        addMessage(`${query} (Focus: ${selectedFocusText})`, 'user');
        queryInput.value = '';
        setProcessingState(true);
        cleanupStatus.textContent = '';
        let payload = {
            session_id: currentSessionId,
            query: query,
            focus_area: focusAreaValue,
            custom_instructions: null
        };
        if (focusAreaValue.startsWith('custom_')) {
            payload.focus_area = 'custom';
            payload.custom_instructions = customPrompts[focusAreaValue];
        }

        try {
            console.log("SUBMITTING QUERY: Sending payload to /query:", payload);
            const response = await fetch('/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            console.log("QUERY RESPONSE: Received response from server with status:", response.status);

            if (!response.ok) {
                let errorMsg = `Server responded with an error: ${response.status}`;
                try {
                     const errData = await response.json();
                     errorMsg = errData.detail || errorMsg;
                } catch (e) { /* Ignore JSON parse error if body is not JSON */ }
                throw new Error(errorMsg);
            }

            const result = await response.json();
            console.log("QUERY RESULT (PARSED JSON):", result);

            if (result && result.answer) {
                console.log("SUCCESS: 'answer' property found. Adding message to chatbox.");
                addMessage(result.answer, 'bot');
            } else {
                console.error("ERROR: Response successful, but the 'answer' property was missing or empty in the JSON result.", result);
                addMessage("Received a response from the server, but it had no content to display.", 'status');
            }
        } catch (error) {
            console.error('QUERY CATCH BLOCK: An error occurred:', error);
            addMessage(`An error occurred while getting the response: ${error.message}`, 'status');
        } finally {
            console.log("QUERY FINALLY BLOCK: Re-enabling controls.");
            setProcessingState(false);
            if (currentSessionId) queryInput.focus();
        }
    }

    sendButton.addEventListener('click', submitQuery);
    queryInput.addEventListener('keypress', (event) => { if (event.key === 'Enter') submitQuery(); });

    endChatButton.addEventListener('click', async () => {
        if (!currentSessionId || isProcessing) return;
        addMessage('Ending session and cleaning up resources...', 'status');
        setProcessingState(true);
        cleanupStatus.textContent = 'Cleaning up...';
        cleanupStatus.style.color = '#666';
        try {
            const response = await fetch('/end-session', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: currentSessionId }),
            });
            const result = await response.json();
            if (response.ok && result.success) {
                cleanupStatus.textContent = '✅ Resources cleaned up successfully.';
                cleanupStatus.style.color = 'green';
                addMessage('Session ended. It is now safe to close the window.', 'status');
                resetUI();
            } else {
                 cleanupStatus.textContent = `❌ Cleanup failed: ${result.message || 'Unknown error'}`;
                 cleanupStatus.style.color = 'red';
                 addMessage(`Error ending session: ${result.message || 'Please try again or refresh.'}`, 'status');
                 setProcessingState(false);
            }
        } catch (error) {
            console.error('End session error:', error);
            cleanupStatus.textContent = '❌ Network or server error during cleanup.';
            cleanupStatus.style.color = 'red';
            addMessage('Network or server error during cleanup. Check console.', 'status');
            setProcessingState(false);
        }
    });

    resetUI();
});