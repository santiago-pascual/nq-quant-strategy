# Security Policy

## Supported Versions

Security issues are currently addressed on the latest version of the `main` branch.

## Reporting a Vulnerability

If you discover a security vulnerability involving this repository, please do not disclose sensitive information publicly through an Issue or Discussion.

Instead, contact the repository maintainer privately through the contact information associated with the repository.

Please include:

- A description of the vulnerability.
- Steps to reproduce it.
- The potential impact.
- Any relevant logs, screenshots, or proof of concept.

## Sensitive Information

Never commit:

- API keys
- Authentication tokens
- Passwords
- Broker credentials
- Exchange credentials
- Private keys
- `.env` files containing secrets

Use environment variables for sensitive configuration.
