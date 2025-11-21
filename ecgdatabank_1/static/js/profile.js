 // State management
let currentSection = 'main';

// DOM elements
const app = document.getElementById('app');
const backBtn = document.getElementById('backBtn');
const pageTitle = document.getElementById('pageTitle');
const pageSubtitle = document.getElementById('pageSubtitle');

// Section titles
const sectionTitles = {
    main: 'Settings',
    profile: 'User Profile',
    profileEdit: 'Edit Profile',
    version: 'Version',
    // theme: 'Dark Theme',
    feedback: 'Send Feedback',
};

// Initialize the app
function init() {
    setupEventListeners();
    // updateThemeUI();
}

// Setup event listeners
function setupEventListeners() {
    // Menu item clicks
    document.querySelectorAll('.menu-item').forEach(item => {
        item.addEventListener('click', () => {
            const section = item.getAttribute('data-section');
            navigateToSection(section);
        });
    });

    // Back button
    backBtn.addEventListener('click', () => {
        navigateToSection('main');
    });

    // Password toggles
    document.querySelectorAll('.password-toggle').forEach(toggle => {
        toggle.addEventListener('click', (e) => {
            const targetId = e.currentTarget.getAttribute('data-target');
            const input = document.getElementById(targetId);
            const eyeOpen = e.currentTarget.querySelector('.eye-open');
            const eyeClosed = e.currentTarget.querySelector('.eye-closed');
            
            if (input.type === 'password') {
                input.type = 'text';
                eyeOpen.classList.add('hidden');
                eyeClosed.classList.remove('hidden');
            } else {
                input.type = 'password';
                eyeOpen.classList.remove('hidden');
                eyeClosed.classList.add('hidden');
            }
        });
    });

    // Profile edit form
    const profileEditForm = document.getElementById('profileEditForm');
    profileEditForm.addEventListener('submit', handleProfileEditSubmit);

}

// Navigation
function navigateToSection(section) {
    // Hide all sections
    document.querySelectorAll('.section').forEach(sec => {
        sec.classList.remove('active');
    });

    // Show target section
    const targetSection = document.getElementById(section === 'main' ? 'mainMenu' : `${section}Section`);
    if (targetSection) {
        targetSection.classList.add('active');
    }

    // Update header
    updateHeader(section);
    
    currentSection = section;
}

function updateHeader(section) {
    if (pageTitle) pageTitle.textContent = sectionTitles[section];

    if (section === 'main') {
        if (backBtn) backBtn.classList.add('hidden');
        if (pageSubtitle) pageSubtitle.classList.remove('hidden');
    } else {
        if (backBtn) backBtn.classList.remove('hidden');
        if (pageSubtitle) pageSubtitle.classList.add('hidden');
    }
}

    // Form submission handler
document.getElementById('passwordChangeForm').addEventListener('submit', async function(e) {
    e.preventDefault();

    const currentPassword = document.getElementById('currentPassword').value.trim();
    const newPassword = document.getElementById('newPassword').value.trim();
    const confirmPassword = document.getElementById('confirmPassword').value.trim();

    // Client-side validation
    if (!currentPassword || !newPassword || !confirmPassword) {
        alertSystem.warning('Warning','All fields are required!');
        return;
    }
    if (newPassword !== confirmPassword) {
        alertSystem.warning('Warning','New passwords do not match!');
        return;
    }

    try {
        const response = await fetch('/auth/change_password/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
            },
            body: JSON.stringify({
                currentPassword,
                newPassword
            })
        });

        const result = await response.json();
        if (response.ok) {
            alertSystem.success('Success', result.message || 'Password changed successfully!');
            document.getElementById('passwordChangeForm').reset();
            navigateToSection('main');
        } else {
            alertSystem.warning('Warning', result.error || 'Password change failed!');
        }
    } catch (error) {
        alertSystem.error('Error','Something went wrong. Please try again later.');
    }
});

function handleProfileEditSubmit(e) {
    e.preventDefault();

    const fullName = document.getElementById('editFullName').value.trim();
    const email = document.getElementById('editEmail').value.trim();
    const phone = document.getElementById('editPhone').value.trim();

    // ?? Get current values from the profile display
    const profileDetails = document.querySelector('#profileSection .profile-details');
    const currentName = profileDetails.children[0].querySelector('p').textContent.trim();
    const currentEmail = profileDetails.children[1].querySelector('p').textContent.trim();
    const currentPhone = profileDetails.children[2].querySelector('p').textContent.trim();

    // ?? Check if nothing changed
    if (fullName === currentName && email === currentEmail && phone === currentPhone) {
        alertSystem.warning('Warning', 'No changes detected. Please modify your details before saving.');
        return;
    }

    // ?? Basic field validation
    if (!fullName || !email || !phone) {
        alertSystem.warning('Warning', 'Please fill in all required fields!');
        return;
    }

    // ?? Validate phone number (must be exactly 10 digits)
    if (!/^\d{10}$/.test(phone)) {
        alertSystem.warning('Warning', 'Phone number must be exactly 10 digits!');
        return;
    }

    // ?? Validate email format
    const emailPattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
    if (!emailPattern.test(email)) {
        alertSystem.warning('Warning', 'Please enter a valid email address!');
        return;
    }

    // ?? Send data to backend
    fetch('/auth/update-profile/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
        },
        body: JSON.stringify({
            username: fullName,
            email: email,
            phone: phone
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // ?? Update the profile view with new data
            profileDetails.children[0].querySelector('p').textContent = fullName;
            profileDetails.children[1].querySelector('p').textContent = email;
            profileDetails.children[2].querySelector('p').textContent = phone;

            alertSystem.success('Success', 'Profile updated successfully!');
            navigateToSection('profile');
        } else {
            alertSystem.error('Error', data.error || 'An error occurred while updating the profile.');
        }
    })
    .catch(() => {
        alertSystem.error('Error', 'Something went wrong. Please try again.');
    });
}



// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', init);
document.addEventListener("DOMContentLoaded", () => {
    const topUpBtn = document.getElementById("topUpBtn");
    const addMoneyModal = document.getElementById("addMoneyModal");
    const closeModalBtn = document.getElementById("closeModalBtn");
    const cancelAddMoney = document.getElementById("cancelAddMoney");
    const confirmAddMoney = document.getElementById("confirmAddMoney");
    const amountButtons = document.querySelectorAll(".amount-btn");
    const customAmountInput = document.getElementById("customAmount");

    // Fetch Django-rendered URL safely from hidden input
    const walletAddMoneyUrl = document.getElementById("walletAddMoneyUrl").value;

    // Open modal
    topUpBtn.onclick = () => addMoneyModal.style.display = "flex";

    // Close modal
    [closeModalBtn, cancelAddMoney].forEach(btn => {
        btn.onclick = () => {
            addMoneyModal.style.display = "none";
            resetModal();
        };
    });

    // Close when clicking outside
    window.onclick = (e) => {
        if (e.target === addMoneyModal) {
            addMoneyModal.style.display = "none";
            resetModal();
        }
    };

    // Select predefined amount
    amountButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            amountButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            customAmountInput.value = btn.textContent.trim();
        });
    });

    // Confirm Add Money (POST to Django)
    confirmAddMoney.addEventListener("click", () => {
        const amount = parseFloat(customAmountInput.value);
        if (!amount || amount < 10) {
            alert("Please enter an amount of at least 10.");
            return;
        }

        // Create form dynamically for POST
        const form = document.createElement("form");
        form.method = "POST";
        form.action = walletAddMoneyUrl; //uses Django URL from hidden input
        form.innerHTML = `
            {% csrf_token %}
            <input type="hidden" name="amount" value="${amount}">
            <input type="hidden" name="name" value="${USER_NAME}">
            <input type="hidden" name="email" value="${USER_EMAIL}">
        `;
        document.body.appendChild(form);
        form.submit();
    });

    // Reset modal
    function resetModal() {
        amountButtons.forEach(b => b.classList.remove("active"));
        customAmountInput.value = "";
    }
});