document.addEventListener('DOMContentLoaded', () => {
    const loginForm = document.getElementById('loginForm');
    const signupForm = document.getElementById('signupForm');

    if (loginForm) {
        loginForm.addEventListener('submit', (event) => {
            event.preventDefault();

            const username = document.getElementById('username').value.trim();
            const password = document.getElementById('password').value.trim();

            if (username === '' || password === '') {
                alert('Please enter both username and password.');
            } else {
                // Here you would typically send the data to a server
                alert(`Logging in with username: ${username}`);
                // loginForm.submit(); // Uncomment to allow form submission
            }
        });
    }

    if (signupForm) {
        signupForm.addEventListener('submit', (event) => {
            event.preventDefault();

            const username = document.getElementById('username').value.trim();
            const email = document.getElementById('email').value.trim();
            const password = document.getElementById('password').value.trim();
            const confirmPassword = document.getElementById('confirm-password').value.trim();

            if (username === '' || email === '' || password === '' || confirmPassword === '') {
                alert('Please fill out all fields.');
                return;
            }

            if (password !== confirmPassword) {
                alert('Passwords do not match.');
                return;
            }

            // Here you would typically send the data to a server
            alert(`Signing up with username: ${username} and email: ${email}`);
            // signupForm.submit(); // Uncomment to allow form submission
        });
    }
});