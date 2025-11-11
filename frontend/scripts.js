// Minimal frontend script for Cable Car Tracking System

// API base (same host)
const API_BASE = '';

// Global state
let capturedPhoto = null;
let peopleData = [];

// --- Global functions ---
window.openRegistrationModal = function() {
    const modal = document.getElementById('registrationModal');
    if (!modal) return console.error('registrationModal not found');
    modal.style.display = 'block';
};

window.closeRegistrationModal = function() {
    const modal = document.getElementById('registrationModal');
    if (!modal) return console.error('registrationModal not found');
    modal.style.display = 'none';
    const form = document.getElementById('registrationForm');
    if (form) form.reset();
    resetPhotoCapture();
};

window.capturePhoto = function() {
    const input = document.getElementById('photoInput');
    if (!input) return console.error('photoInput not found');
    input.click();
};

function resetPhotoCapture() {
    capturedPhoto = null;
    const preview = document.getElementById('photoPreview');
    if (!preview) return;
    preview.innerHTML = '<span class="photo-icon">👤</span><p>No photo captured</p>';
}

// Handle file upload
function handlePhotoUpload(event) {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function(e) {
        capturedPhoto = e.target.result; // base64
        const preview = document.getElementById('photoPreview');
        if (preview) preview.innerHTML = `<img src="${capturedPhoto}" alt="Captured photo" style="max-width:100%;height:auto;border-radius:6px;"/>`;
    };
    reader.readAsDataURL(file);
}

// Show simple notification (console + optional visual)
function showNotification(msg, type='info') {
    console.log('Notification:', type, msg);
    // non-blocking toast could be implemented here; keep simple
}

// Load registered people and render table
async function loadPeopleData() {
    try {
        const res = await fetch(`${API_BASE}/api/people`);
        if (!res.ok) return console.error('Failed to load people');
        const data = await res.json();
        peopleData = data.people || [];
        renderPeopleTable();
        // update counters
        const totalRegistered = document.getElementById('totalRegistered');
        const totalIn = document.getElementById('totalIn');
        const totalOut = document.getElementById('totalOut');
        if (totalRegistered) totalRegistered.textContent = peopleData.length;
        if (totalIn) totalIn.textContent = peopleData.filter(p=>p.status==='IN').length;
        if (totalOut) totalOut.textContent = peopleData.filter(p=>p.status==='OUT').length;
    } catch (e) {
        console.error('loadPeopleData error', e);
    }
}

function renderPeopleTable() {
    const tbody = document.getElementById('peopleTableBody');
    if (!tbody) return;
    tbody.innerHTML = '';
    peopleData.forEach(person => {
        const tr = document.createElement('tr');
        const photoTd = document.createElement('td');
        const img = document.createElement('img');
        img.src = person.photo_url || '';
        img.alt = person.name || '';
        img.style.width = '60px'; img.style.height = '60px'; img.style.objectFit = 'cover'; img.style.borderRadius = '6px';
        photoTd.appendChild(img);
        tr.appendChild(photoTd);

        const addCell = (v) => { const td = document.createElement('td'); td.textContent = v || ''; return td; };
        tr.appendChild(addCell(person.name));
        tr.appendChild(addCell(person.age));
        tr.appendChild(addCell(person.location));
        tr.appendChild(addCell(person.status));
        tr.appendChild(addCell(person.last_in || ''));
        tr.appendChild(addCell(person.last_out || ''));
        
        // Actions column with dropdown
        const actionsTd = document.createElement('td');
        const dropdown = document.createElement('div');
        dropdown.className = 'action-dropdown';
        
        const actionBtn = document.createElement('button');
        actionBtn.textContent = '⋯';
        actionBtn.className = 'action-btn';
        actionBtn.title = 'Actions';
        actionBtn.onclick = (e) => toggleDropdown(e, dropdown);
        
        const dropdownContent = document.createElement('div');
        dropdownContent.className = 'dropdown-content';
        
        // Generate Embedding option
        const generateBtn = document.createElement('button');
        generateBtn.innerHTML = '🧠 Generate Embedding';
        generateBtn.className = 'primary';
        generateBtn.onclick = () => {
            generateEmbedding(person.id, person.name);
            hideDropdown(dropdown);
        };
        
        // Test Detection option
        const testBtn = document.createElement('button');
        testBtn.innerHTML = '🎯 Test Detection';
        testBtn.className = 'primary';
        testBtn.onclick = () => {
            testDetection(person.id, person.name);
            hideDropdown(dropdown);
        };
        
        // Delete option
        const deleteBtn = document.createElement('button');
        deleteBtn.innerHTML = '🗑️ Delete Registration';
        deleteBtn.className = 'danger';
        deleteBtn.onclick = () => {
            deletePerson(person.id, person.name);
            hideDropdown(dropdown);
        };
        
        dropdownContent.appendChild(generateBtn);
        dropdownContent.appendChild(testBtn);
        dropdownContent.appendChild(deleteBtn);
        dropdown.appendChild(actionBtn);
        dropdown.appendChild(dropdownContent);
        actionsTd.appendChild(dropdown);
        tr.appendChild(actionsTd);
        
        tbody.appendChild(tr);
    });
}

// Dropdown functions
function toggleDropdown(event, dropdown) {
    event.stopPropagation();
    const dropdownContent = dropdown.querySelector('.dropdown-content');
    
    // Close all other dropdowns first
    document.querySelectorAll('.dropdown-content.show').forEach(d => {
        if (d !== dropdownContent) d.classList.remove('show');
    });
    
    dropdownContent.classList.toggle('show');
}

function hideDropdown(dropdown) {
    const dropdownContent = dropdown.querySelector('.dropdown-content');
    dropdownContent.classList.remove('show');
}

// Close dropdowns when clicking outside
document.addEventListener('click', function(event) {
    if (!event.target.matches('.action-btn')) {
        document.querySelectorAll('.dropdown-content.show').forEach(d => {
            d.classList.remove('show');
        });
    }
});

// Registration handler
async function handleRegistration(event) {
    event.preventDefault();
    try {
        const name = (document.getElementById('personName') || {}).value || '';
        const age = parseInt((document.getElementById('personAge') || {}).value || 0);
        const location = (document.getElementById('personLocation') || {}).value || '';
        if (!name || !age || !location) { showNotification('Please fill all fields','error'); return; }
        if (!capturedPhoto) { showNotification('Please upload/select a photo','error'); return; }

        const submitBtn = event.target.querySelector('button[type="submit"]');
        if (submitBtn) { submitBtn.disabled = true; submitBtn.textContent = 'Registering...'; }

        const payload = { name, age, location, photo: capturedPhoto };
        const res = await fetch(`${API_BASE}/api/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (res.ok && data.success) {
            showNotification(data.message || 'Registered','success');
            window.closeRegistrationModal();
            await loadPeopleData();
        } else {
            showNotification(data.message || 'Registration failed','error');
            console.error('Registration error', data);
        }
        if (submitBtn) { submitBtn.disabled = false; submitBtn.textContent = 'Register Person'; }
    } catch (e) {
        console.error('handleRegistration error', e);
        showNotification('Registration error','error');
        const submitBtn = event.target.querySelector('button[type="submit"]');
        if (submitBtn) { submitBtn.disabled = false; submitBtn.textContent = 'Register Person'; }
    }
}

// Delete person function
async function deletePerson(personId, personName) {
    if (!confirm(`Are you sure you want to delete ${personName}? This will remove all data including images and embeddings.`)) {
        return;
    }
    
    try {
        const res = await fetch(`${API_BASE}/api/people/${personId}/delete`, {
            method: 'DELETE'
        });
        
        const data = await res.json();
        
        if (res.ok && data.success) {
            showNotification(data.message || 'Person deleted successfully', 'success');
            await loadPeopleData(); // Refresh the table
        } else {
            showNotification(data.message || 'Failed to delete person', 'error');
        }
    } catch (error) {
        console.error('Delete error:', error);
        showNotification('Error deleting person', 'error');
    }
}

// Generate embedding function
async function generateEmbedding(personId, personName) {
    if (!confirm(`Generate face embedding for ${personName}? This will process their uploaded image for face recognition.`)) {
        return;
    }
    
    try {
        // Show loading state
        const generateBtn = event.target;
        const originalText = generateBtn.textContent;
        generateBtn.textContent = '⏳ Processing...';
        generateBtn.disabled = true;
        
        const res = await fetch(`${API_BASE}/api/people/${personId}/generate-embedding`, {
            method: 'POST'
        });
        
        const data = await res.json();
        
        if (res.ok && data.success) {
            showNotification(data.message || 'Face embedding generated successfully', 'success');
            console.log('Embedding details:', data.details);
        } else {
            showNotification(data.message || 'Failed to generate embedding', 'error');
        }
        
        // Restore button state
        generateBtn.textContent = originalText;
        generateBtn.disabled = false;
        
    } catch (error) {
        console.error('Generate embedding error:', error);
        showNotification('Error generating embedding', 'error');
        
        // Restore button state
        const generateBtn = event.target;
        generateBtn.textContent = '🧠 Generate';
        generateBtn.disabled = false;
    }
}

// Test detection function
async function testDetection(personId, personName) {
    const cameraType = prompt(`Test detection for ${personName}:\nEnter camera type (in/out):`, 'in');
    
    if (!cameraType || (cameraType !== 'in' && cameraType !== 'out')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/people/${personId}/test-detection`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                camera_type: cameraType
            })
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            showNotification(`${data.message} for ${personName}`, 'success');
            await loadPeopleData(); // Reload the table to show updated status
        } else {
            showNotification(data.message || 'Test detection failed', 'error');
        }
    } catch (error) {
        console.error('Test detection error:', error);
        showNotification('Error testing detection', 'error');
    }
}

// Check if person has embedding
async function checkEmbedding(personId) {
    try {
        const res = await fetch(`${API_BASE}/api/people/${personId}/check-embedding`);
        const data = await res.json();
        return data.has_embedding;
    } catch (error) {
        console.error('Check embedding error:', error);
        return false;
    }
}

// Export CSV function
async function exportToCSV() {
    try {
        const res = await fetch(`${API_BASE}/api/export-csv`);
        if (res.ok) {
            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'tracking_data.csv';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            showNotification('CSV exported successfully', 'success');
        } else {
            showNotification('Failed to export CSV', 'error');
        }
    } catch (error) {
        console.error('Export error:', error);
        showNotification('Error exporting CSV', 'error');
    }
}

// Real-time detection polling
let detectionInterval;

function startRealTimeUpdates() {
    // Poll for updates every 2 seconds
    detectionInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}/api/detections/recent`);
            if (response.ok) {
                const data = await response.json();
                if (data.success) {
                    updatePeopleStatusRealTime(data.people_status);
                }
            }
        } catch (error) {
            console.log('Real-time update error:', error);
        }
    }, 2000);
}

function stopRealTimeUpdates() {
    if (detectionInterval) {
        clearInterval(detectionInterval);
        detectionInterval = null;
    }
}

function updatePeopleStatusRealTime(peopleStatus) {
    // Update the people data with new status
    let statusChanged = false;
    
    peopleData.forEach(person => {
        if (peopleStatus[person.id] && peopleStatus[person.id].status !== person.status) {
            person.status = peopleStatus[person.id].status;
            statusChanged = true;
        }
    });
    
    if (statusChanged) {
        renderPeopleTable();
        updateStatusCards();
    }
}

function updateStatusCards() {
    const totalRegistered = document.getElementById('totalRegistered');
    const totalIn = document.getElementById('totalIn');
    const totalOut = document.getElementById('totalOut');
    
    if (totalRegistered) totalRegistered.textContent = peopleData.length;
    if (totalIn) totalIn.textContent = peopleData.filter(p => p.status === 'IN').length;
    if (totalOut) totalOut.textContent = peopleData.filter(p => p.status === 'OUT').length;
}

// Setup event listeners
function setupEventListeners() {
    const photoInput = document.getElementById('photoInput');
    if (photoInput) photoInput.addEventListener('change', handlePhotoUpload);
    const form = document.getElementById('registrationForm');
    if (form) form.addEventListener('submit', handleRegistration);

    // Close modal when clicking outside
    window.addEventListener('click', function(e){
        const modal = document.getElementById('registrationModal');
        if (!modal) return;
        if (e.target === modal) modal.style.display = 'none';
    });

    // Search input
    const search = document.getElementById('searchInput');
    if (search) search.addEventListener('input', function(){
        const q = this.value.toLowerCase();
        const filtered = peopleData.filter(p => (p.name||'').toLowerCase().includes(q));
        // render filtered list
        const tbody = document.getElementById('peopleTableBody');
        if (!tbody) return;
        tbody.innerHTML = '';
        filtered.forEach(person => {
            const tr = document.createElement('tr');
            const photoTd = document.createElement('td');
            const img = document.createElement('img');
            img.src = person.photo_url || '';
            img.alt = person.name || '';
            img.style.width = '60px'; img.style.height = '60px'; img.style.objectFit = 'cover'; img.style.borderRadius = '6px';
            photoTd.appendChild(img);
            tr.appendChild(photoTd);
            const addCell = (v) => { const td = document.createElement('td'); td.textContent = v || ''; return td; };
            tr.appendChild(addCell(person.name));
            tr.appendChild(addCell(person.age));
            tr.appendChild(addCell(person.location));
            tr.appendChild(addCell(person.status));
            tr.appendChild(addCell(person.last_in || ''));
            tr.appendChild(addCell(person.last_out || ''));
            tbody.appendChild(tr);
        });
    });
}

// Initialize
document.addEventListener('DOMContentLoaded', function(){
    setupEventListeners();
    loadPeopleData();
    startRealTimeUpdates();
});

// Expose functions for debugging
window.loadPeopleData = loadPeopleData;
window.handleRegistration = handleRegistration;
window.startRealTimeUpdates = startRealTimeUpdates;
window.stopRealTimeUpdates = stopRealTimeUpdates;