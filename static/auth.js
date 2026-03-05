function clearAuthFields() {
    const ids = [
        "loginEmail",
        "loginPassword",
        "adminEmail",
        "adminPassword",
        "regName",
        "regEmail",
        "regPassword",
        "forgotEmail",
        "forgotOtp",
        "forgotNewPassword"
    ];
    ids.forEach((id) => {
        const el = document.getElementById(id);
        if (el) el.value = "";
    });
}

// LOGIN
function login() {
    fetch("/login", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            email: document.getElementById("loginEmail").value,
            password: document.getElementById("loginPassword").value
        })
    })
    .then(res => res.json())
    .then(data => {
    if (data.status === "success") {
        clearAuthFields();
        window.location.href = data.redirect_url || "/dashboard";
    } else {
        alert(data.message);
    }
})

    .catch(err => console.error(err));
}

// ADMIN LOGIN
function adminLogin() {
    fetch("/admin/login", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            email: document.getElementById("adminEmail").value,
            password: document.getElementById("adminPassword").value
        })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === "success") {
            clearAuthFields();
            window.location.href = data.redirect_url || "/admin";
        } else {
            alert(data.message);
        }
    })
    .catch(err => console.error(err));
}

// REGISTER
function register() {
    fetch("/register", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            name: document.getElementById("regName").value,
            email: document.getElementById("regEmail").value,
            password: document.getElementById("regPassword").value
        })
    })
    .then(res => res.json())
    .then(data => {
        if (data.email_error) {
            alert(`${data.message}\n\nError: ${data.email_error}`);
        } else {
            alert(data.message);
        }
        clearAuthFields();
    })
    .catch(err => console.error(err));
}

// FORGOT PASSWORD - SEND OTP
function forgotPassword() {
    const email = (document.getElementById("forgotEmail")?.value || "").trim();
    if (!email) {
        alert("Please enter your registered email.");
        return;
    }

    fetch("/auth/forgot/request", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ email })
    })
    .then(res => res.json())
    .then(data => alert(data.message || "Request completed"))
    .catch(err => console.error(err));
}

// FORGOT PASSWORD - VERIFY OTP + RESET
function resetPasswordWithOtp() {
    const email = (document.getElementById("forgotEmail")?.value || "").trim();
    const otp = (document.getElementById("forgotOtp")?.value || "").trim();
    const newPassword = document.getElementById("forgotNewPassword")?.value || "";

    if (!email || !otp || !newPassword) {
        alert("Email, OTP, and new password are required.");
        return;
    }

    fetch("/auth/forgot/reset", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            email: email,
            otp: otp,
            new_password: newPassword
        })
    })
    .then(res => res.json())
    .then(data => {
        alert(data.message || "Request completed");
        if (data.status === "success") {
            const forgotOtp = document.getElementById("forgotOtp");
            const forgotNewPassword = document.getElementById("forgotNewPassword");
            if (forgotOtp) forgotOtp.value = "";
            if (forgotNewPassword) forgotNewPassword.value = "";
        }
    })
    .catch(err => console.error(err));
}

// CONTACT FORM
function submitForm() {
    const nameEl = document.getElementById("name");
    const emailEl = document.getElementById("email");
    const messageEl = document.getElementById("message");

    fetch("/contact", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            name: nameEl ? nameEl.value : "",
            email: emailEl ? emailEl.value : "",
            message: messageEl ? messageEl.value : ""
        })
    })
    .then(res => res.json())
    .then(data => {
        alert(data.message);
        if (nameEl) nameEl.value = "";
        if (emailEl) emailEl.value = "";
        if (messageEl) messageEl.value = "";
    })
    .catch(err => console.error(err));
}
function logout() {
    fetch("/logout")
    .then(res => res.json())
    .then(data => {
        alert(data.message);
        window.location.reload(); // Refresh page
    })
    .catch(err => console.error(err));
}  

document.addEventListener("DOMContentLoaded", () => {
    clearAuthFields();
    const authModal = document.getElementById("authModal");
    if (authModal) {
        authModal.addEventListener("show.bs.modal", clearAuthFields);
    }
});

window.addEventListener("pageshow", clearAuthFields);


