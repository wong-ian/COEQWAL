// static/app.js
document.addEventListener('DOMContentLoaded', () => {
    const uploadButton = document.getElementById('upload-button');
    const fileInput = document.getElementById('document-upload');
    const uploadStatus = document.getElementById('upload-status');
    const chatbox = document.getElementById('chatbox');
    const queryInput = document.getElementById('query-input');
    const sendButton = document.getElementById('send-button');
    const thinkingIndicator = document.getElementById('thinking-indicator');
    // --- New elements ---
    const endChatButton = document.getElementById('end-chat-button');
    const cleanupStatus = document.getElementById('cleanup-status');

    let currentSessionId = null; // Store the session ID received after successful upload
    let isProcessing = false; // Flag to prevent multiple submissions

    // Function to add messages to the chatbox
    function addMessage(content, type = 'status') {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');

        const contentDiv = document.createElement('div');
        contentDiv.classList.add('content');
        // Use textContent initially to prevent XSS, then replace newlines for bot messages
        contentDiv.textContent = content;

        if (type === 'user') {
            messageDiv.classList.add('user-message');
        } else if (type === 'bot') {
            messageDiv.classList.add('bot-message');
            // Basic Markdown-like formatting for newlines (replace \n with <br>)
            contentDiv.innerHTML = contentDiv.innerHTML.replace(/\n/g, '<br>');
        } else { // status message
            messageDiv.classList.add('status-message');
            // Status messages usually don't need the inner content div styling
             messageDiv.textContent = content; // Direct text content for status
        }

        if (type !== 'status') {
             messageDiv.appendChild(contentDiv);
        }

        chatbox.appendChild(messageDiv);
        chatbox.scrollTop = chatbox.scrollHeight; // Scroll to bottom
    }

    // Function to update UI state (disable/enable inputs)
    function setProcessingState(processing) {
        isProcessing = processing;
        queryInput.disabled = processing || !currentSessionId;
        sendButton.disabled = processing || !currentSessionId;
        uploadButton.disabled = processing;
        fileInput.disabled = processing;
        endChatButton.disabled = processing || !currentSessionId; // Disable end chat during processing
        thinkingIndicator.style.display = processing ? 'block' : 'none';
        if (!currentSessionId) { // Always disable end chat if no session
             endChatButton.disabled = true;
        }
    }

    // Function to reset UI to initial state after ending session
    function resetUI() {
        currentSessionId = null;
        chatbox.innerHTML = '<div class="status-message">Please upload a document to begin analysis.</div>';
        uploadStatus.textContent = '';
        cleanupStatus.textContent = ''; // Clear cleanup status
        fileInput.value = ''; // Clear the file input selection
        setProcessingState(false); // Re-enable upload, disable others
    }

    // Handle file upload
    uploadButton.addEventListener('click', async () => {
        const file = fileInput.files[0];
        if (!file) {
            uploadStatus.textContent = 'Please select a file first.';
            uploadStatus.style.color = 'red';
            return;
        }

        // Reset UI before starting new upload if a session exists
        if (currentSessionId) {
             addMessage('Starting new upload, previous session data will be cleared on success...', 'status');
             // Don't reset UI fully yet, wait for successful upload
        } else {
             chatbox.innerHTML = ''; // Clear chat only if no session existed
        }

        addMessage(`Uploading "${file.name}"...`, 'status');
        uploadStatus.textContent = `Uploading "${file.name}"...`;
        uploadStatus.style.color = '#666';
        cleanupStatus.textContent = ''; // Clear previous cleanup status
        setProcessingState(true);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData,
            });

            const result = await response.json();

            if (response.ok && result.success) {
                // Server automatically cleaned up old session resources associated with the *cookie* session ID
                // Now store the *potentially new* session ID from the response
                currentSessionId = result.session_id;
                uploadStatus.textContent = `✅ Ready: "${result.filename}" (Session Active)`;
                uploadStatus.style.color = 'green';
                // Clear chat only on successful upload of the *first* document in a browser lifecycle,
                // or leave messages if it was a re-upload. Let's clear it for simplicity now.
                chatbox.innerHTML = '';
                addMessage(`Document "${result.filename}" processed. You can now ask questions.`, 'status');
                setProcessingState(false); // Enable query input and end chat button
                queryInput.focus();
            } else {
                uploadStatus.textContent = `❌ Error: ${result.message || 'Upload failed'}`;
                uploadStatus.style.color = 'red';
                addMessage(`Error processing document: ${result.message || 'Please try again.'}`, 'status');
                // Don't reset session ID here, backend might have kept it if upload failed but session existed
                setProcessingState(false); // Re-enable upload but not query/end chat if session is uncertain
                // Check if session ID still exists from cookie if needed, but safer to disable query/end
                queryInput.disabled = true;
                sendButton.disabled = true;
                endChatButton.disabled = true;
            }
        } catch (error) {
            console.error('Upload error:', error);
            uploadStatus.textContent = '❌ Network or server error during upload.';
            uploadStatus.style.color = 'red';
             addMessage('Network or server error during upload. Check console.', 'status');
            // Keep potential existing session ID? Or clear? Let's clear for safety.
            // currentSessionId = null; // Let's not clear, maybe server kept session
            setProcessingState(false);
             queryInput.disabled = true; // Keep disabled on major error
             sendButton.disabled = true;
             endChatButton.disabled = true;
        }
    });

    // Handle query submission
    async function submitQuery() {
        const query = queryInput.value.trim();
        if (!query || !currentSessionId || isProcessing) {
            return;
        }

        addMessage(query, 'user');
        queryInput.value = ''; // Clear input field
        setProcessingState(true);
        cleanupStatus.textContent = ''; // Clear cleanup status

        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ session_id: currentSessionId, query: query }),
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

            // Log local sources, but maybe don't display in chat unless desired
            if (result.local_sources && result.local_sources.length > 0) {
                console.log("Local sources retrieved:", result.local_sources);
            }

        } catch (error) {
            console.error('Query error:', error);
            addMessage(`Error getting response: ${error.message || 'Check console.'}`, 'status');
        } finally {
            setProcessingState(false);
            queryInput.focus();
        }
    }

    sendButton.addEventListener('click', submitQuery);
    queryInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            submitQuery();
        }
    });

    // --- Handle End Chat Button ---
    endChatButton.addEventListener('click', async () => {
        if (!currentSessionId || isProcessing) {
            return;
        }

        // Optional: Confirm with the user
        // if (!confirm("Are you sure you want to end the chat? This will delete your uploaded document from the server.")) {
        //     return;
        // }

        addMessage('Ending session and cleaning up resources...', 'status');
        setProcessingState(true); // Disable everything while cleaning up
        cleanupStatus.textContent = 'Cleaning up...';
        cleanupStatus.style.color = '#666';


        try {
            const response = await fetch('/end-session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                // Send the current session ID to the backend
                body: JSON.stringify({ session_id: currentSessionId }),
            });

            const result = await response.json(); // Expect { success: boolean, message: string }

            if (response.ok && result.success) {
                cleanupStatus.textContent = '✅ Resources cleaned up successfully.';
                cleanupStatus.style.color = 'green';
                addMessage('Session ended. It is now safe to close the window.', 'status');
                resetUI(); // Reset the entire UI
            } else {
                 cleanupStatus.textContent = `❌ Cleanup failed: ${result.message || 'Unknown error'}`;
                 cleanupStatus.style.color = 'red';
                 addMessage(`Error ending session: ${result.message || 'Please try again or refresh.'}`, 'status');
                 setProcessingState(false); // Re-enable controls? Or leave disabled? Let's re-enable.
                 // Keep currentSessionId so they *could* try again? Or clear? Let's keep it for retry.
            }

        } catch (error) {
            console.error('End session error:', error);
            cleanupStatus.textContent = '❌ Network or server error during cleanup.';
            cleanupStatus.style.color = 'red';
            addMessage('Network or server error during cleanup. Check console.', 'status');
            setProcessingState(false); // Re-enable controls
        }
        // Note: We don't call setProcessingState(false) on SUCCESS because resetUI() does it.
    });


    // Initial state
    resetUI(); // Call resetUI on load to set initial state correctly

});