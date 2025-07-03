// static/app.js
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

    console.log("app.js: DOMContentLoaded - Script loaded.");

    let currentSessionId = null;
    let isProcessing = false;

    function addMessage(content, type = 'status') {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        const contentDiv = document.createElement('div');
        contentDiv.classList.add('content');
        contentDiv.textContent = content; // Safely set text content

        if (type === 'user') {
            messageDiv.classList.add('user-message');
        } else if (type === 'bot') {
            messageDiv.classList.add('bot-message');
            // For bot messages, replace newlines with <br> for display
            contentDiv.innerHTML = contentDiv.innerHTML.replace(/\n/g, '<br>');
        } else { // status message
            messageDiv.classList.add('status-message');
            // Status messages typically don't need the inner styled 'content' div
            messageDiv.textContent = content; // Set text directly on the messageDiv
        }

        if (type !== 'status') { // Only append styled contentDiv for user/bot messages
             messageDiv.appendChild(contentDiv);
        }

        chatbox.appendChild(messageDiv);
        chatbox.scrollTop = chatbox.scrollHeight; // Scroll to bottom
    }

    function setProcessingState(processing) {
        console.log(`app.js: setProcessingState CALLED. processing: ${processing}, currentSessionId: ${currentSessionId}`);
        isProcessing = processing;

        const sessionIsActive = !!currentSessionId; // True if session ID exists, false otherwise
        const canInteract = !processing && sessionIsActive; // User can interact if not processing AND session is active

        // Query-related controls
        queryInput.disabled = !canInteract;
        sendButton.disabled = !canInteract;
        analysisFocusSelect.disabled = !canInteract;
        console.log(`app.js: setProcessingState - analysisFocusSelect.disabled set to: ${analysisFocusSelect.disabled}`);

        // Upload controls are disabled only during an active processing step
        uploadButton.disabled = processing;
        fileInput.disabled = processing;

        // End chat button is enabled if a session is active, but disabled during other processing
        endChatButton.disabled = processing || !sessionIsActive;

        thinkingIndicator.style.display = processing ? 'block' : 'none';
    }

    function resetUI() {
        console.log("app.js: resetUI CALLED.");
        currentSessionId = null;
        chatbox.innerHTML = '<div class="status-message">Please upload a document and select an analysis focus to begin.</div>';
        uploadStatus.textContent = '';
        cleanupStatus.textContent = '';
        fileInput.value = ''; // Clear the file input
        analysisFocusSelect.value = 'general'; // Reset dropdown to default
        setProcessingState(false); // This will correctly disable query/focus controls as currentSessionId is null
    }

    uploadButton.addEventListener('click', async () => {
        console.log("app.js: uploadButton CLICKED.");
        const file = fileInput.files[0];
        if (!file) {
            uploadStatus.textContent = 'Please select a file first.';
            uploadStatus.style.color = 'red';
            return;
        }

        // If a session already exists, a new upload implies clearing the old session server-side.
        // The UI will reflect this after a successful new upload.
        if (currentSessionId) {
             addMessage('Uploading new document. Previous session resources will be replaced on success...', 'status');
        } else {
             chatbox.innerHTML = ''; // Clear for a completely new session
        }

        addMessage(`Uploading "${file.name}"...`, 'status');
        uploadStatus.textContent = `Uploading "${file.name}"...`;
        uploadStatus.style.color = '#666';
        cleanupStatus.textContent = ''; // Clear any previous cleanup messages
        setProcessingState(true); // Disable controls, show spinner

        const formData = new FormData();
        formData.append('file', file);

        try {
            console.log("app.js: uploadButton - Attempting fetch('/upload').");
            const response = await fetch('/upload', { method: 'POST', body: formData });
            console.log("app.js: uploadButton - fetch('/upload') response status:", response.status);

            const result = await response.json();
            console.log("app.js: uploadButton - fetch('/upload') result:", result);

            if (response.ok && result.success) {
                console.log("app.js: uploadButton - Upload successful.");
                currentSessionId = result.session_id; // IMPORTANT: Update session ID
                uploadStatus.textContent = `✅ Ready: "${result.filename}" (Session Active)`;
                uploadStatus.style.color = 'green';
                if (chatbox.innerHTML.includes('Please upload a document')) { // Only clear if it was initial message
                    chatbox.innerHTML = '';
                }
                addMessage(`Document "${result.filename}" processed. Select focus and ask questions.`, 'status');
                setProcessingState(false); // Re-evaluate and enable controls
                queryInput.focus();
            } else {
                console.warn("app.js: uploadButton - Upload failed or response not ok.", result);
                uploadStatus.textContent = `❌ Error: ${result.message || 'Upload failed'}`;
                uploadStatus.style.color = 'red';
                addMessage(`Error processing document: ${result.message || 'Please try again.'}`, 'status');
                // Don't clear currentSessionId here, as the backend might still hold it if the cookie persists
                // and a subsequent action might rely on it. The UI should reflect that interaction is not possible.
                setProcessingState(false); // Reset processing flag
                // Explicitly disable interactive controls again if upload failed
                queryInput.disabled = true;
                sendButton.disabled = true;
                analysisFocusSelect.disabled = true;
                // endChatButton state handled by setProcessingState based on currentSessionId
            }
        } catch (error) {
            console.error('app.js: uploadButton - fetch CATCH block error:', error);
            uploadStatus.textContent = '❌ Network or server error during upload.';
            uploadStatus.style.color = 'red';
            addMessage('Network or server error during upload. Check console.', 'status');
            setProcessingState(false); // Reset processing flag
            // Explicitly disable interactive controls on major error
            queryInput.disabled = true;
            sendButton.disabled = true;
            analysisFocusSelect.disabled = true;
        }
    });

    async function submitQuery() {
        console.log("app.js: submitQuery CALLED.");
        const query = queryInput.value.trim();
        const focusArea = analysisFocusSelect.value;

        if (!query || !currentSessionId || isProcessing || !focusArea) {
            console.log(`app.js: submitQuery - Aborting. Query: ${!!query}, SessionID: ${!!currentSessionId}, Processing: ${isProcessing}, FocusArea: ${focusArea}`);
            if (!focusArea && currentSessionId) {
                addMessage('Please select an analysis focus from the dropdown.', 'status');
            }
             else if (!currentSessionId) {
                addMessage('Please upload a document first.', 'status');
            }
            return;
        }

        const selectedFocusText = analysisFocusSelect.options[analysisFocusSelect.selectedIndex].text;
        addMessage(`${query} (Focus: ${selectedFocusText})`, 'user');

        queryInput.value = '';
        setProcessingState(true);
        cleanupStatus.textContent = '';

        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: currentSessionId,
                    query: query,
                    focus_area: focusArea
                }),
            });

            if (!response.ok) {
                let errorMsg = `HTTP error ${response.status}`;
                try {
                     const errData = await response.json();
                     errorMsg = errData.detail || errorMsg;
                } catch(e) { /* Ignore JSON parse error */ }
                throw new Error(errorMsg);
            }

            const result = await response.json();
            addMessage(result.answer, 'bot');

            if (result.local_sources && result.local_sources.length > 0) {
                console.log("Local sources retrieved:", result.local_sources);
            }

        } catch (error) {
            console.error('Query error:', error);
            addMessage(`Error getting response: ${error.message || 'Check console.'}`, 'status');
        } finally {
            setProcessingState(false);
            if (currentSessionId) queryInput.focus(); // Only focus if session is still active
        }
    }

    sendButton.addEventListener('click', submitQuery);
    queryInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            submitQuery();
        }
    });

    endChatButton.addEventListener('click', async () => {
        console.log("app.js: endChatButton CLICKED.");
        if (!currentSessionId || isProcessing) {
            return;
        }

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
                resetUI(); // This will set currentSessionId to null and call setProcessingState
            } else {
                 cleanupStatus.textContent = `❌ Cleanup failed: ${result.message || 'Unknown error'}`;
                 cleanupStatus.style.color = 'red';
                 addMessage(`Error ending session: ${result.message || 'Please try again or refresh.'}`, 'status');
                 setProcessingState(false); // Re-enable controls if cleanup failed but session might still be considered active
            }
        } catch (error) {
            console.error('End session error:', error);
            cleanupStatus.textContent = '❌ Network or server error during cleanup.';
            cleanupStatus.style.color = 'red';
            addMessage('Network or server error during cleanup. Check console.', 'status');
            setProcessingState(false);
        }
    });

    console.log("app.js: Adding initial event listeners and calling resetUI.");
    resetUI(); // Call resetUI on load to set initial state correctly
    console.log("app.js: Initial resetUI call complete.");
});