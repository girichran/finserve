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
    })
    .catch(err => console.error(err));
}

// CONTACT FORM
function submitForm() {
    fetch("/contact", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            name: document.getElementById("name").value,
            email: document.getElementById("email").value,
            message: document.getElementById("message").value
        })
    })
    .then(res => res.json())
    .then(data => alert(data.message))
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


