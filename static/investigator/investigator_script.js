// This file controls the Investigator's Manage Cases dashboard interactivity.

// Helper function: Shared API call for status update (Defined Globally)
async function updateStatusApi(caseId, status, publicNote = null, privateInstructions = null) {
    const payload = { status: status };
    if (publicNote) payload.halt_message = publicNote;
    if (privateInstructions) payload.halt_instructions = privateInstructions;

    try {
        const resp = await fetch(`/api/investigator/update_status/${caseId}`, {
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body: JSON.stringify(payload)
        });
        const json = await resp.json();
        
        if (json.success) {
             window.location.reload();
        } else {
             alert(json.error || 'Failed to update status.');
        }
    } catch(err){
        console.error("Status update failed:", err);
        alert('Request failed.');
    }
}

// Helper function: View Case Details (FIXED for async/await and Blank Modal)
async function viewCase(caseId) {
    const modal = document.getElementById("case-details-modal");
    
    // 1. Set loading state immediately
    if (modal) modal.style.display = "block";
    
    // NOTE: Populate loading status for better UX
    document.getElementById("modal-tracking-id-display").textContent = "Loading...";
    document.getElementById("modal-db-id-display").textContent = "Loading...";


    try {
        // Fetch detailed report data from the API
        const response = await fetch(`/api/investigator/get_report_details/${caseId}`);
        const data = await response.json();

        if (data.error) {
            alert("Error fetching details: " + data.error);
            if (modal) modal.style.display = "none";
            return;
        }

        // 2. Populate modal fields (Use data response)
      // 2. Populate modal fields (Use data response)
            document.getElementById("modal-report-id-header").textContent = data.report_tracking_id || "N/A";
            document.getElementById("modal-tracking-id-display").textContent = data.report_tracking_id || "N/A";
            document.getElementById("modal-db-id-display").textContent = data._id || "N/A";

            // Reporter Info
            document.getElementById("modal-reporter-name").textContent = data.name || "N/A";
            document.getElementById("modal-reporter-email").textContent = data.email || "N/A";
            document.getElementById("modal-reporter-mobile").textContent = data.mobile || "N/A";
            document.getElementById("modal-reporter-gender").textContent = data.gender || "N/A";
            document.getElementById("modal-reporter-age").textContent = data.age || "N/A";

            // Case Info
            document.getElementById("modal-case-status").textContent = data.status || "N/A";
            document.getElementById("modal-predicted-category").textContent = data.predicted_category || "N/A";
            document.getElementById("modal-predicted-urgency").textContent = data.predicted_urgency || "N/A";
            document.getElementById("modal-state").textContent = data.state || "N/A";
            document.getElementById("modal-location").textContent = data.location || "N/A";
            document.getElementById("modal-date-of-incident").textContent = data.date_of_incident || "N/A";
            document.getElementById("modal-date-submitted").textContent = data.date_submitted || "N/A";
            document.getElementById("modal-case-description").textContent = data.description || "N/A";

        
        // Evidence Link (FINAL FIX: Use custom /uploads/ route)
        const evidenceLinkEl = document.getElementById("inv-modal-evidence-link");
        if (data.evidence_file && data.evidence_file !== "N/A") {
            const evidenceUrl = `/uploads/${data.evidence_file}`; 
            evidenceLinkEl.innerHTML = `<a href="${evidenceUrl}" target="_blank" class="btn btn-sm btn-info">View Evidence</a>`;
        } else {
            evidenceLinkEl.textContent = "No File";
        }

        // Final Report Link (FINAL FIX: Use custom /uploads/ route)
        const reportLinkEl = document.getElementById("inv-modal-final-report-link");
        if (data.final_report_file && data.final_report_file !== "N/A") {
            const reportUrl = `/uploads/${data.final_report_file}`;
            reportLinkEl.innerHTML = `<a href="${reportUrl}" target="_blank">Download Report</a>`;
        } else {
            reportLinkEl.textContent = "Not Uploaded";
        }
        
        // NOTE: You must ensure all other modal fields (status, category, etc.) are populated 
        // using logic similar to the 'modal-reporter-name' lines above.

    }
    catch (err) {
        console.error("Error fetching case details:", err);
        alert("Failed to fetch case details.");
        if (modal) modal.style.display = "none";
    }
}


document.addEventListener('DOMContentLoaded', function() {
    
    // =========================================================
    // === MODAL ELEMENT DEFINITIONS (For Global Access) =======
    // =========================================================
    const caseDetailsModal = document.getElementById('case-details-modal');
    const uploadReportModal = document.getElementById('uploadReportModal');
    const investigatorHaltModal = document.getElementById('investigator-halt-modal');
    
    // Halt Modal Fields
    const haltMessageInput = document.getElementById('inv_halt_message');
    const haltInstructionsInput = document.getElementById('inv_halt_instructions');
    const haltCaseIdInput = document.getElementById('haltCaseId');


    // =========================================================
    // === 1. MODAL CLOSE LISTENERS (Fixes the broken 'X' button)
    // =========================================================
    
    // Close Case Details Modal ('X' and outside click)
    if (caseDetailsModal) {
        const closeModalBtn = caseDetailsModal.querySelector('.close-button') || caseDetailsModal.querySelector('span[onclick]');
        
        if (closeModalBtn) {
            closeModalBtn.addEventListener('click', () => {
                caseDetailsModal.style.display = 'none';
            });
        }
        window.addEventListener('click', (event) => {
            if (event.target === caseDetailsModal) {
                caseDetailsModal.style.display = 'none';
            }
        });
    }

    // Close Halt Modal ('X' button)
    document.querySelector('.close-halt-modal')?.addEventListener('click', () => {
        if (investigatorHaltModal) investigatorHaltModal.style.display = 'none';
    });
    
    // --- FINAL FIX: Close Upload Report Modal ('X' button) ---
    const closeUploadModalBtn = uploadReportModal ? uploadReportModal.querySelector('.close-button') : null;

    if (closeUploadModalBtn) {
        closeUploadModalBtn.addEventListener('click', () => {
            if (uploadReportModal) uploadReportModal.style.display = 'none';
            document.getElementById('uploadReportForm').reset();
        });
    }
    window.addEventListener('click', (event) => {
        if (event.target === uploadReportModal) {
            uploadReportModal.style.display = 'none';
            document.getElementById('uploadReportForm').reset();
        }
    });
    // ----------------------------------------------------------


    // =========================================================
    // === 2. TABLE BUTTON EVENT LISTENERS =======================
    // =========================================================

    // Event delegation for all buttons inside the table body
    document.querySelectorAll(".cases-table tbody").forEach(tbody => {
        tbody.addEventListener('click', async (event) => {
            const target = event.target.closest('button');
            if (!target) return;

            const row = target.closest('tr');
            const caseId = row.dataset.caseId;
            const dbId = target.dataset.dbId; // Used by View button
            
            // --- View Button Logic ---
            if (target.classList.contains('view-case-btn')) {
                if (dbId) viewCase(dbId);
                return;
            }

            // --- Final Report Button Logic (Opens Upload Modal) ---
            if (target.classList.contains('open-report-modal-btn')) {
                // Ensure the modal case ID is the tracking ID from the row
                document.getElementById('modalCaseId').value = caseId;
                if (uploadReportModal) uploadReportModal.style.display = 'block';
                return;
            }

            // --- Halt/Update Button Logic (Handles action from Select dropdown) ---
            if (target.classList.contains('update-status-btn')) {
                const select = row.querySelector('.status-select');
                const newStatus = select ? select.value : '';

                if (!newStatus || newStatus === "") { 
                    alert('Please select an action from the dropdown first.');
                    return; 
                }
                
                // If the user chooses to Halt the case, open the dedicated modal
                if (newStatus === 'Halted') {
                    if (haltCaseIdInput) haltCaseIdInput.value = caseId;
                    if (haltMessageInput) haltMessageInput.value = ''; // Clear fields
                    if (haltInstructionsInput) haltInstructionsInput.value = '';
                    if (investigatorHaltModal) investigatorHaltModal.style.display = 'block';
                    return;
                }

                // If user chooses 'Received' or 'In Progress' (Regular status change)
                await updateStatusApi(caseId, newStatus);
                return;
            }
            
            // --- Resume Button Logic (For cases already Halted) ---
            if (target.classList.contains('resume-btn')) {
                await updateStatusApi(caseId, 'In Progress');
                return;
            }
        });
    });
    
    
    // =========================================================
    // === 3. HALT MODAL SUBMISSION HANDLER ======================
    // =========================================================
    
    document.getElementById('invSubmitHalt')?.addEventListener('click', async function(){
        const caseId = haltCaseIdInput ? haltCaseIdInput.value : '';
        
        const publicNote = haltMessageInput ? haltMessageInput.value.trim() : ''; 
        const privateInstructions = haltInstructionsInput ? haltInstructionsInput.value.trim() : ''; 

        if (!publicNote) { 
            alert("Please provide a reason in the 'PUBLIC NOTE' box to inform the user why the case is being put on hold.");
            return;
        }

        // Send payload using the custom update API helper
        await updateStatusApi(caseId, 'Halted', publicNote, privateInstructions);
        
        // Close the modal (API helper handles the reload if successful)
        if (investigatorHaltModal) investigatorHaltModal.style.display = 'none';
    });


    // =========================================================
    // === 4. FINAL REPORT UPLOAD SUBMISSION (Keep original logic)
    // =========================================================
    
    const form = document.getElementById('uploadReportForm');
    const modalCaseIdInput = document.getElementById('modalCaseId');

    if (form) {
        form.addEventListener('submit', function(event) {
            event.preventDefault();
            const caseId = modalCaseIdInput.value;
            const fileInput = document.getElementById('final_report_file');
            
            if (fileInput.files.length === 0) {
                alert('Please select a file to upload.');
                return;
            }

            const formData = new FormData();
            formData.append('final_report_file', fileInput.files[0]);

            fetch(`/api/investigator/upload_report/${caseId}`, {
                method: 'POST',
                body: formData 
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert(data.message);
                    if (uploadReportModal) uploadReportModal.style.display = 'none';
                    window.location.reload(); 
                } else {
                    alert('Error: ' + data.error);
                }
            })
            .catch(error => console.error('Error uploading report:', error));
        });
    }

});