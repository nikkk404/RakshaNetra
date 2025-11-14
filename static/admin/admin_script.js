document.addEventListener('DOMContentLoaded', () => {
    console.log("admin_script.js loaded. DOMContentLoaded event fired.");

    // Helper function for capitalization (good for display in UI)
    function capitalize(str) {
        if (typeof str !== 'string' || str.length === 0) {
            return '';
        }
        return str.charAt(0).toUpperCase() + str.slice(1);
    }

    // --- Main Dispatcher ---
    function initializeAdminPageSpecifics() {
        if (document.getElementById('manage-cases')) {
            initializeManageCasesPage();
        }
        if (document.getElementById('reports')) {
            initializeReportsPage();
        }
        console.log("Admin page-specific JS initialization complete.");
    }
// --- New function to handle the Workload Modal ---
function initializeInvestigatorWorkload() {
    // NOTE: Ensure your HTML contains the 'workload-details-modal' with the correct IDs

    document.querySelectorAll('.investigator-row').forEach(row => {
        row.addEventListener('click', async function() {
            const investigatorId = this.dataset.investigatorId;
            const username = this.cells[0].textContent.trim();
            const pendingCountEl = this.cells[4].querySelector('.badge'); // Assuming pending count is in the 5th cell
            const pendingCount = pendingCountEl ? pendingCountEl.textContent.trim() : '0';

            // Elements for the modal that MUST be defined in your HTML
            const modalTitle = document.getElementById('workload-modal-title');
            const modalBody = document.getElementById('workload-modal-body');
            const workloadModal = document.getElementById('workload-details-modal'); 

            if (!workloadModal || !modalTitle || !modalBody) {
                console.error("Workload modal elements not found.");
                return;
            }

            if (parseInt(pendingCount) === 0) {
                alert(`${username} has no active pending cases.`);
                return;
            }

            modalTitle.textContent = `${username}'s Pending Workload (${pendingCount} Cases)`;
            modalBody.innerHTML = '<div class="text-center py-4"><i class="fas fa-spinner fa-spin me-2"></i> Loading case details...</div>';
            workloadModal.style.display = 'block';

            try {
                // Calls the Flask API you defined: /api/admin/investigator_pending_cases/
                const response = await fetch(`/api/admin/investigator_pending_cases/${investigatorId}`);
                const cases = await response.json();

                let html = '<ul class="list-group">';
                
                if (cases.length === 0) {
                     html = '<div class="alert alert-info">No pending cases found matching status criteria.</div>';
                } else {
                    cases.forEach(caseItem => {
                        const urgencyClass = `urgency-${caseItem.predicted_urgency.toLowerCase()}`;
                        const haltReason = caseItem.halt_message ? `<p class="text-danger small mt-1">Halted Reason: ${caseItem.halt_message}</p>` : '';
                        const rejectionReason = caseItem.admin_rejection_reason ? `<p class="text-danger small mt-1">Rejected: ${caseItem.admin_rejection_reason}</p>` : '';

                        html += `
                            <li class="list-group-item d-flex justify-content-between align-items-start ${urgencyClass}">
                                <div>
                                    <strong>Case ID: ${caseItem.report_tracking_id}</strong>
                                    <span class="badge bg-secondary ms-2">${caseItem.status}</span>
                                    <span class="badge ${urgencyClass}">${caseItem.predicted_urgency}</span>
                                    <p class="mb-0 small text-muted">Desc: ${caseItem.description.substring(0, 70)}...</p>
                                    ${haltReason}
                                    ${rejectionReason}
                                </div>
                                <button class="btn btn-sm btn-info view-details-link" data-db-id="${caseItem._id_str}">View</button>
                            </li>
                        `;
                    });
                    html += '</ul>';
                }
                
                modalBody.innerHTML = html;

            } catch (error) {
                console.error("Error fetching workload details:", error);
                modalBody.innerHTML = '<div class="alert alert-danger">Failed to load case details. Please check console.</div>';
            }
        });
    });
}

// Ensure this function is called at the end of the script:
// initializeInvestigatorWorkload();
    // --- Manage Cases Page Specific JS (Interactivity for table and modal) ---
    function initializeManageCasesPage() {
        console.log("Initializing Manage Cases Page JS for interactive table and modal.");

        const casesTableBody = document.getElementById('cases-table-body');
        const caseDetailsModal = document.getElementById('case-details-modal');
        const searchInput = document.getElementById('case-search');
        const statusFilter = document.getElementById('case-status-filter');
        const applyFilterBtn = document.getElementById('apply-filter-btn');

        // Elements for the Final Report Upload Modal
        const finalReportUploadModal = document.getElementById('final-report-upload-modal');
        const finalReportIncidentDbIdInput = finalReportUploadModal ? document.getElementById('final-report-incident-db-id') : null;
        const finalReportModalCaseIdSpan = finalReportUploadModal ? document.getElementById('final-report-modal-case-id') : null;
        
        // Modal elements fetched once for performance and safety
        const newStatusSelect = document.getElementById('new_status');
        const publicNoteInput = document.getElementById('public_note');
        const haltMessageInput = document.getElementById('halt_message');
        const haltInstructionsInput = document.getElementById('halt_instructions');
        const haltFields = document.getElementById('halt-fields');
        const statusModal = document.getElementById('status-modal'); // Fetch status modal globally
        const closeStatusModalBtn = document.querySelector('.close-status-modal'); // Fetch close button

        // Function to filter table rows based on search and status
        function filterTableRows() {
            const searchTerm = searchInput ? searchInput.value.toLowerCase().trim() : '';
            const selectedStatus = statusFilter ? statusFilter.value.toLowerCase() : 'all';

            const rows = casesTableBody ? casesTableBody.querySelectorAll('tr') : [];
            
            rows.forEach(row => {
                const reportId = row.cells[0]?.textContent?.toLowerCase() || ''; 
                const descriptionSnippet = row.cells[1]?.textContent?.toLowerCase() || ''; 
                const status = row.cells[2]?.querySelector('span')?.textContent?.toLowerCase() || ''; 
                const predictedCategory = row.cells[3]?.textContent?.toLowerCase() || ''; 
                const predictedUrgency = row.cells[4]?.textContent?.toLowerCase() || '';

                const matchesSearch = reportId.includes(searchTerm) || 
                                      descriptionSnippet.includes(searchTerm) ||
                                      predictedCategory.includes(searchTerm) ||
                                      predictedUrgency.includes(searchTerm);

                const matchesStatus = selectedStatus === 'all' || status === selectedStatus;

                if (matchesSearch && matchesStatus) {
                    row.style.display = ''; 
                } else {
                    row.style.display = 'none'; 
                }
            });
            console.log("Table filtering applied.");
        }

        // Event listeners for filtering
        if (applyFilterBtn) applyFilterBtn.addEventListener('click', filterTableRows);
        if (searchInput) searchInput.addEventListener('input', filterTableRows); 
        if (statusFilter) statusFilter.addEventListener('change', filterTableRows); 


        // Event delegation for table action buttons (View, Update, Resolve)
        if (casesTableBody) { 
            casesTableBody.addEventListener('click', async (event) => {
                const target = event.target.closest('button'); 
                if (!target) return; 

                const reportTrackingId = target.closest('tr')?.cells[0]?.textContent?.trim();
                const mongoDbId = target.dataset.dbId; 

                console.log(`Action: "${target.textContent.trim()}" on DB ID: ${mongoDbId}`);
                
                // --- View button logic (Original logic to fetch/show large details modal) ---
                if (target.classList.contains('view-case-btn')) {
                    // NOTE: Your original logic for fetching/populating the caseDetailsModal goes here
                    if (caseDetailsModal) caseDetailsModal.style.display = 'block';
                    console.log("View Case button clicked and modal shown.");
                } 
                
                // --- FIXED: Update button logic (Opens Status Modal directly) ---
                else if (target.classList.contains('update-case-btn')) {
                    
                    // 1. Set the case ID into the modal form
                    document.getElementById('statusCaseId').value = mongoDbId;
                    
                    // 2. Clear previous status/notes (SAFELY HANDLING NULL CHECK)
                    if (newStatusSelect) newStatusSelect.value = ""; 
                    if (publicNoteInput) publicNoteInput.value = ""; 
                    if (haltMessageInput) haltMessageInput.value = ""; 
                    if (haltInstructionsInput) haltInstructionsInput.value = ""; 
                    
                    // 3. Hide halt fields and show the modal IMMEDIATELY
                    if (haltFields) haltFields.style.display = 'none';
                    if (statusModal) statusModal.style.display = 'block'; 
                    
                    console.log(`Status modal opened for ID: ${mongoDbId}`);
                } 

                // --- Mark Resolved button logic (Direct action) ---
                else if (target.classList.contains('mark-resolved-btn')) {
                    const confirmResolve = confirm(`Are you sure you want to mark Report ID ${reportTrackingId} as Resolved? This will finalize the case.`);
                    if (!confirmResolve) return; 
                    
                    try {
                        const response = await fetch(`/api/admin/update_case_status/${mongoDbId}`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ status: 'Resolved' }) 
                        });
                        const result = await response.json(); 

                        if (response.ok) {
                            alert(result.message);
                            window.location.reload(); 
                        } else {
                            console.error("API resolve failed:", result.error || response.statusText);
                            alert(`Failed to mark as resolved. ` + (result.details || result.message || ""));
                        }
                    } catch (error) {
                        console.error("Error during resolve fetch:", error);
                        alert("An error occurred during resolve action.");
                    }
                }
            });
        }
    } // End initializeManageCasesPage
    

    // --- Reports Page Specific JS (Remains conceptual) ---
    function initializeReportsPage() {
        console.log("Initializing Reports Page JS (conceptual functions).");
        // ... (Remains conceptual) ...
    }
    
    // --- Overall Initialization Dispatcher ---
    initializeAdminPageSpecifics(); 
    
    
    // ===================================================================
    // === ASSIGN / REVOKE LOGIC (RESTORED) ==============================
    // ===================================================================
    
    // 🔹 Assign Investigator
    document.querySelectorAll(".assign-btn").forEach((btn) => {
        btn.addEventListener("click", async () => {
            const caseId = btn.getAttribute("data-case-id");
            const selectEl = document.querySelector(`.investigator-select[data-case-id="${caseId}"]`);
            const investigatorId = selectEl ? selectEl.value : "";

            if (!investigatorId) {
                alert("Please select an investigator before assigning.");
                return;
            }

            try {
                // NOTE: This route auto-sets status to "In Progress" in the Flask backend
                const response = await fetch(`/admin/assign_investigator/${caseId}`, { 
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ investigator_id: investigatorId })
                });

                if (response.ok) {
                    alert("Investigator assigned successfully! Status is now 'In Progress'.");
                    window.location.reload(); 
                } else {
                    const err = await response.json();
                    alert("Error assigning investigator: " + (err.message || response.statusText));
                }
            } catch (err) {
                console.error("Error assigning investigator:", err);
                alert("An error occurred while assigning investigator.");
            }
        });
    });

    // 🔹 Revoke Investigator
    document.querySelectorAll(".revoke-btn").forEach((btn) => {
        btn.addEventListener("click", async () => {
            const caseId = btn.getAttribute("data-case-id");

            if (!confirm("Are you sure you want to revoke this investigator?")) return;

            try {
                const response = await fetch(`/admin/revoke_investigator/${caseId}`, {
                    method: "POST"
                });

                if (response.ok) {
                    alert("Investigator revoked successfully!");
                    window.location.reload();
                } else {
                    const err = await response.json();
                    alert("Error revoking investigator: " + (err.message || response.statusText));
                }
            } catch (err) {
                console.error("Error revoking investigator:", err);
                alert("An error occurred while revoking investigator.");
            }
        });
    });
    
    // ===================================================================
    // === STATUS MODAL LISTENERS (Submission Handler) =====================
    // ===================================================================
    
    const statusForm = document.getElementById('statusUpdateForm');
    
    if (statusForm) {
        statusForm.addEventListener('submit', async function(e){
            e.preventDefault();
            const caseId = document.getElementById('statusCaseId').value;
            const newStatusSelect = document.getElementById('new_status');
            const status = newStatusSelect.value;
            
            // Read the public_note field
            const publicNote = document.getElementById('public_note')?.value; 
            
            const payload = {status: status};

            // Add the public note to the payload for ALL status changes
            if (publicNote) {
                payload.public_note = publicNote; 
            }
            
            // Add halt fields only if Halted status is selected
            if (status === 'Halted') {
                payload.halt_message = document.getElementById('halt_message').value;
                payload.halt_instructions = document.getElementById('halt_instructions').value;
            }
            
            try {
                const resp = await fetch(`/api/admin/update_case_status/${caseId}`, {
                    method: 'POST',
                    headers:{'Content-Type':'application/json'},
                    body: JSON.stringify(payload)
                });
                const json = await resp.json();
                if (json.success) {
                    alert(json.message || 'Status updated');
                    
                    // FIX: Redirect to the CLEAN manage_cases route to avoid URL query parameter pollution
                    window.location.href = '/admin/manage_cases'; 
                } else {
                    alert(json.error || json.message || 'Failed to update');
                }
            } catch(err){
                alert('Request failed');
            }
        });
    }
    
    // --- FIX: Add Close Modal Listener (Targeting the X button and outside click) ---
    const statusModal = document.getElementById('status-modal');
    if (statusModal) {
        const closeStatusModalBtn = document.querySelector('.close-status-modal');
        
        if (closeStatusModalBtn) {
            closeStatusModalBtn.addEventListener('click', () => statusModal.style.display = 'none');
        }
        
        // Close modal when clicking outside
        window.addEventListener('click', (event) => {
            if (event.target === statusModal) {
                statusModal.style.display = 'none';
            }
        });
        
        // Handle Halted status visibility toggle (since this was defined inside initializeManageCasesPage before)
        const newStatusSelect = document.getElementById('new_status');
        const haltFields = document.getElementById('halt-fields');

        if (newStatusSelect && haltFields) {
            newStatusSelect.addEventListener('change', function(){
                if (this.value === 'Halted') {
                    haltFields.style.display = 'block';
                } else {
                    haltFields.style.display = 'none';
                }
            });
        }
    }
    // --------------------------------------------------------------------------------


    
}); // End main DOMContentLoaded

// ===============================================
// ✅ CASE DETAILS DATA LOADER (Required Function)
// ===============================================
async function populateCaseDetailsModal(caseId) {
    try {
        const response = await fetch(`/api/admin/get_case_details/${caseId}`);
        const data = await response.json();

        if (!data.success) {
            alert("Failed to load details: " + data.error);
            return;
        }

        const c = data.case;

        document.getElementById("modal-report-id-header").innerText = c.report_tracking_id || "N/A";
        document.getElementById("modal-tracking-id-display").innerText = c.report_tracking_id || "N/A";
        document.getElementById("modal-db-id-display").innerText = c._id || "N/A";
        document.getElementById("modal-report-type").innerText = c.registration_type || "N/A";
        document.getElementById("modal-reporter-name").innerText = c.name || "Anonymous";
        document.getElementById("modal-reporter-email").innerText = c.email || "N/A";
        document.getElementById("modal-reporter-mobile").innerText = c.mobile || "N/A";
        document.getElementById("modal-reporter-gender").innerText = c.gender || "N/A";
        document.getElementById("modal-reporter-age").innerText = c.age || "N/A";
        document.getElementById("modal-case-status").innerText = c.status || "N/A";
        document.getElementById("modal-predicted-category").innerText = c.predicted_category || "N/A";
        document.getElementById("modal-predicted-urgency").innerText = c.predicted_urgency || "N/A";
        document.getElementById("modal-state").innerText = c.state || "N/A";
        document.getElementById("modal-location").innerText = c.location || "N/A";
        document.getElementById("modal-date-of-incident").innerText = c.date_of_incident || "N/A";
        document.getElementById("modal-date-submitted").innerText = c.date_submitted || "N/A";
        document.getElementById("modal-case-description").innerText = c.description || "N/A";

                // ===================== EVIDENCE FILE LINK =====================
            if (c.evidence_file) {
                document.getElementById("modal-evidence-link").innerHTML =
                    `<a href="/uploads/${c.evidence_file}" target="_blank" class="btn btn-primary btn-sm">View Evidence</a>`;
            } else {
                document.getElementById("modal-evidence-link").innerText = "Not Available";
            }


            // ===================== FINAL REPORT FILE LINK =====================
            if (c.final_report_file) {
                document.getElementById("modal-final-report-link").innerHTML =
                    `<a href="/uploads/${c.final_report_file}" target="_blank" class="btn btn-success btn-sm">Download Final Report</a>`;
            } else {
                document.getElementById("modal-final-report-link").innerText = "Not Available";
            }


            // ✅ Show modal
            document.getElementById("case-details-modal").style.display = "block";


    } catch (error) {
        console.error("Error loading case details:", error);
        alert("Unexpected error. Check console.");
    }
}
