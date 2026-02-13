"""Tests for redactor.py — sensitive data detection and reversible masking."""

import sys
from pathlib import Path

import pytest

from redactor import Redactor, Redaction


class TestIPv4Detection:
    """Test IPv4 address detection."""

    def test_single_ipv4(self):
        r = Redactor()
        text, redactions = r.redact("Connect to 192.168.1.50 on port 22")
        assert "192.168.1.50" not in text
        assert "[IP_1]" in text
        assert len(redactions) == 1
        assert redactions[0].original == "192.168.1.50"

    def test_multiple_ipv4(self):
        r = Redactor()
        text, redactions = r.redact("From 10.0.0.1 to 10.0.0.2")
        assert "10.0.0.1" not in text
        assert "10.0.0.2" not in text
        ip_redactions = [rd for rd in redactions if rd.category == "ipv4"]
        assert len(ip_redactions) == 2

    def test_consistent_placeholder(self):
        r = Redactor()
        text, _ = r.redact("Server 192.168.1.1 and again 192.168.1.1")
        # Same IP should get same placeholder
        assert text.count("[IP_1]") == 2

    def test_public_ip(self):
        r = Redactor()
        text, redactions = r.redact("Reached 24.34.33.12 via route")
        assert "24.34.33.12" not in text
        assert len(redactions) == 1


class TestFQDNDetection:
    """Test hostname/FQDN detection."""

    def test_fqdn_with_domain(self):
        r = Redactor()
        text, redactions = r.redact("Can't reach prod-db-west.aws")
        assert "prod-db-west.aws" not in text
        fqdn_r = [rd for rd in redactions if rd.category == "fqdn"]
        assert len(fqdn_r) == 1

    def test_complex_fqdn(self):
        r = Redactor()
        text, _ = r.redact("Check api.staging.internal")
        assert "api.staging.internal" not in text

    def test_multiple_fqdns(self):
        r = Redactor()
        text, _ = r.redact("From web.corp to db.corp")
        assert "web.corp" not in text
        assert "db.corp" not in text


class TestUUIDDetection:
    """Test UUID detection."""

    def test_standard_uuid(self):
        r = Redactor()
        uuid = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        text, redactions = r.redact(f"Resource ID: {uuid}")
        assert uuid not in text
        assert "[UUID_1]" in text

    def test_uppercase_uuid(self):
        r = Redactor()
        uuid = "A1B2C3D4-E5F6-7890-ABCD-EF1234567890"
        text, _ = r.redact(f"ID: {uuid}")
        assert uuid not in text


class TestEmailDetection:
    """Test email address detection."""

    def test_simple_email(self):
        r = Redactor()
        text, redactions = r.redact("Contact josh@company.com for help")
        assert "josh@company.com" not in text
        assert "[EMAIL_1]" in text

    def test_complex_email(self):
        r = Redactor()
        text, _ = r.redact("Send to admin+tag@sub.domain.co.uk")
        assert "admin+tag@sub.domain.co.uk" not in text


class TestURLDetection:
    """Test URL detection."""

    def test_http_url(self):
        r = Redactor()
        text, redactions = r.redact("Visit http://myserver.internal:8080/api/v1")
        url_r = [rd for rd in redactions if rd.category == "url"]
        assert len(url_r) >= 1

    def test_https_url(self):
        r = Redactor()
        text, _ = r.redact("Endpoint: https://api.mycompany.com/v2/data")
        assert "api.mycompany.com" not in text


class TestAPIKeyDetection:
    """Test API key / token detection."""

    def test_openai_key(self):
        r = Redactor()
        key = "sk-abc123def456ghi789jkl012mno345pq"
        text, redactions = r.redact(f"Using key {key}")
        assert key not in text

    def test_github_token(self):
        r = Redactor()
        token = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl"
        text, _ = r.redact(f"Token: {token}")
        assert token not in text

    def test_aws_access_key(self):
        r = Redactor()
        key = "AKIAIOSFODNN7EXAMPLE"
        text, _ = r.redact(f"AWS key: {key}")
        assert key not in text


class TestAWSResourceDetection:
    """Test AWS resource ID detection."""

    def test_instance_id(self):
        r = Redactor()
        text, _ = r.redact("Instance i-0abc123def456789a is down")
        assert "i-0abc123def456789a" not in text

    def test_security_group(self):
        r = Redactor()
        text, _ = r.redact("SG: sg-12345678")
        assert "sg-12345678" not in text

    def test_vpc_id(self):
        r = Redactor()
        text, _ = r.redact("VPC vpc-abcdef12 has no IGW")
        assert "vpc-abcdef12" not in text


class TestIPv6Detection:
    """Test IPv6 address detection."""

    def test_link_local_with_zone(self):
        r = Redactor()
        text, redactions = r.redact(
            "inet6 fe80::1802:ddbd:b93:5ba3%en0 prefixlen 64"
        )
        assert "1802:ddbd:b93:5ba3" not in text
        ipv6_r = [rd for rd in redactions if rd.category == "ipv6"]
        assert len(ipv6_r) >= 1

    def test_full_ipv6(self):
        r = Redactor()
        addr = "2600:4040:44ad:600:a:53e6:3dd8:3727"
        text, _ = r.redact(f"inet6 {addr} prefixlen 64")
        assert addr not in text

    def test_compressed_ipv6(self):
        r = Redactor()
        text, _ = r.redact("addr ::ffff:192.168.1.1 mapped")
        # ::ffff prefix should be caught
        assert "::ffff" not in text or "192.168.1.1" not in text


class TestMACAddressDetection:
    """Test MAC address detection."""

    def test_colon_separated(self):
        r = Redactor()
        text, _ = r.redact("MAC: 00:1A:2B:3C:4D:5E")
        assert "00:1A:2B:3C:4D:5E" not in text

    def test_dash_separated(self):
        r = Redactor()
        text, _ = r.redact("MAC: 00-1A-2B-3C-4D-5E")
        assert "00-1A-2B-3C-4D-5E" not in text

    def test_macos_short_form(self):
        """macOS uses single-digit hex octets like 0:e0:4c:68:7:cf."""
        r = Redactor()
        text, _ = r.redact("joshs-air (192.168.1.163) at 0:e0:4c:68:7:cf on en0")
        assert "0:e0:4c:68:7:cf" not in text

    def test_macos_short_form_2(self):
        r = Redactor()
        text, _ = r.redact("at a2:0:ac:d8:fd:39 on en0 ifscope [ethernet]")
        assert "a2:0:ac:d8:fd:39" not in text

    def test_macos_multicast(self):
        r = Redactor()
        text, _ = r.redact("mdns.mcast.net (224.0.0.251) at 1:0:5e:0:0:fb on en0")
        assert "1:0:5e:0:0:fb" not in text


class TestFilePathDetection:
    """Test file path with username detection."""

    def test_linux_home_path(self):
        r = Redactor()
        text, _ = r.redact("File at /home/josh/project/config.yaml")
        assert "/home/josh" not in text

    def test_macos_users_path(self):
        r = Redactor()
        text, _ = r.redact("Found in /Users/josh/Documents/secrets.txt")
        assert "/Users/josh" not in text


class TestCIDRDetection:
    """Test CIDR notation detection."""

    def test_cidr_block(self):
        r = Redactor()
        text, _ = r.redact("Subnet 10.0.0.0/24 is full")
        assert "10.0.0.0/24" not in text


class TestConnectionStringDetection:
    """Test connection string detection."""

    def test_postgres_url(self):
        r = Redactor()
        text, _ = r.redact("DB: postgresql://user:pass@db.internal:5432/mydb")
        assert "postgresql://user:pass@db.internal:5432/mydb" not in text

    def test_mongodb_url(self):
        r = Redactor()
        text, _ = r.redact("Mongo: mongodb+srv://admin:secret@cluster.mongodb.net")
        assert "mongodb+srv://admin:secret@cluster.mongodb.net" not in text


class TestUserDictionary:
    """Test user-defined redaction terms."""

    def test_user_term_applied(self):
        r = Redactor(user_terms={"acme-corp": "[COMPANY]"})
        text, redactions = r.redact("Working at acme-corp on internal tools")
        assert "acme-corp" not in text
        assert "[COMPANY]" in text
        assert any(rd.category == "user_term" for rd in redactions)

    def test_user_term_priority(self):
        """User terms should be applied even if they don't match regex."""
        r = Redactor(user_terms={"Project Falcon": "[PROJECT]"})
        text, _ = r.redact("Progress on Project Falcon is good")
        assert "Project Falcon" not in text
        assert "[PROJECT]" in text

    def test_multiple_user_terms(self):
        r = Redactor(user_terms={
            "acme-corp": "[COMPANY]",
            "prod-west": "[CLUSTER]",
        })
        text, _ = r.redact("acme-corp prod-west cluster")
        assert "acme-corp" not in text
        assert "prod-west" not in text


class TestReversibility:
    """Test that redaction is properly reversible."""

    def test_unredact_simple(self):
        r = Redactor()
        text, _ = r.redact("Server 192.168.1.50 on port 22")
        restored = r.unredact(text)
        assert "192.168.1.50" in restored

    def test_unredact_multiple(self):
        r = Redactor()
        original = "From 10.0.0.1 to josh@example.com"
        text, _ = r.redact(original)
        restored = r.unredact(text)
        assert "10.0.0.1" in restored
        assert "josh@example.com" in restored

    def test_unredact_preserves_non_redacted(self):
        r = Redactor()
        text, _ = r.redact("Hello world on port 22")
        restored = r.unredact(text)
        assert "Hello world on port 22" == restored

    def test_unredact_user_terms(self):
        r = Redactor(user_terms={"acme-corp": "[COMPANY]"})
        text, _ = r.redact("Working at acme-corp")
        restored = r.unredact(text)
        assert "acme-corp" in restored


class TestConsistentMapping:
    """Test that the same value always gets the same placeholder."""

    def test_same_ip_across_calls(self):
        r = Redactor()
        text1, _ = r.redact("Server 10.0.0.1 is up")
        text2, _ = r.redact("Ping 10.0.0.1 works")
        # Both should use [IP_1]
        assert "[IP_1]" in text1
        assert "[IP_1]" in text2

    def test_different_ips_different_placeholders(self):
        r = Redactor()
        text, _ = r.redact("10.0.0.1 and 10.0.0.2")
        assert "[IP_1]" in text
        assert "[IP_2]" in text


class TestManualRedaction:
    """Test manual add/remove of redactions."""

    def test_add_manual_redaction(self):
        r = Redactor()
        text, redaction = r.add_manual_redaction("secret-project", "Working on secret-project today")
        assert "secret-project" not in text
        assert redaction.auto is False

    def test_remove_redaction(self):
        r = Redactor()
        text, redactions = r.redact("Server 192.168.1.1 is fine")
        assert "[IP_1]" in text

        restored = r.remove_redaction(redactions[0], text)
        assert "192.168.1.1" in restored


class TestNoFalsePositives:
    """Test that common text is NOT incorrectly redacted."""

    def test_plain_text_no_redactions(self):
        r = Redactor()
        text, redactions = r.redact("Hello, how are you today?")
        assert text == "Hello, how are you today?"
        assert len(redactions) == 0

    def test_numbers_not_redacted(self):
        r = Redactor()
        text, redactions = r.redact("I have 42 items in 3 boxes")
        assert text == "I have 42 items in 3 boxes"
        assert len(redactions) == 0

    def test_port_numbers_not_redacted(self):
        r = Redactor()
        text, _ = r.redact("Port 443 and port 8080 are open")
        assert "443" in text
        assert "8080" in text

    def test_code_snippet_preserved(self):
        r = Redactor()
        text, _ = r.redact("Use `print('hello')` in Python")
        assert "print('hello')" in text


class TestCombinedScenario:
    """Test realistic combined scenarios."""

    def test_network_troubleshooting(self):
        r = Redactor()
        original = "Why can't 24.34.33.12 reach prod-db-west.aws via port 22?"
        text, redactions = r.redact(original)
        assert "24.34.33.12" not in text
        assert "prod-db-west.aws" not in text
        assert "port 22" in text  # Port should NOT be redacted
        assert len(redactions) == 2

        # Verify unredact works
        restored = r.unredact(text)
        assert "24.34.33.12" in restored
        assert "prod-db-west.aws" in restored

    def test_aws_infrastructure(self):
        r = Redactor()
        original = (
            "Instance i-0abc123def456789a in vpc-abcdef12 "
            "can't reach sg-12345678"
        )
        text, redactions = r.redact(original)
        assert "i-0abc123def456789a" not in text
        assert "vpc-abcdef12" not in text
        assert "sg-12345678" not in text
        assert len(redactions) >= 3  # 3 AWS + possible hex overlaps

    def test_mixed_sensitive_data(self):
        r = Redactor(user_terms={"acme-corp": "[COMPANY]"})
        original = (
            "User josh@acme-corp.com on 10.0.0.5 reported that "
            "the acme-corp staging server api.staging.internal is down"
        )
        text, redactions = r.redact(original)
        assert "josh@acme-corp.com" not in text
        assert "10.0.0.5" not in text
        assert "acme-corp" not in text
        assert "api.staging.internal" not in text
        assert len(redactions) >= 4


class TestMappingTable:
    """Test mapping table export."""

    def test_empty_mapping(self):
        r = Redactor()
        assert r.get_mapping_table() == []

    def test_populated_mapping(self):
        r = Redactor()
        r.redact("Server 10.0.0.1 is up")
        table = r.get_mapping_table()
        assert len(table) == 1
        assert table[0][0] == "10.0.0.1"
        assert table[0][1] == "[IP_1]"


# ── New CLI-Specific Pattern Tests ──────────────────────────────


class TestSSHPublicKeyDetection:
    """Test SSH public key blob detection."""

    def test_ed25519_key(self):
        r = Redactor()
        line = "192.168.1.172 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPGYCsXI3FCI212zg6DivpHf12P6067gHR9ID8KL5H7u"
        text, redactions = r.redact(line)
        key_r = [rd for rd in redactions if rd.category == "ssh_pubkey"]
        assert len(key_r) == 1
        assert "AAAAC3NzaC1lZDI1NTE5" not in text
        assert "[SSHKEY_1]" in text

    def test_rsa_key(self):
        r = Redactor()
        line = "host ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC03UxrpiKmhrCNiHo2k6"
        text, _ = r.redact(line)
        assert "AAAAB3NzaC1yc2E" not in text

    def test_ecdsa_key(self):
        r = Redactor()
        line = "host ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTY"
        text, _ = r.redact(line)
        assert "AAAAE2VjZHNhLXNo" not in text

    def test_known_hosts_full_line(self):
        r = Redactor()
        line = "github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl"
        text, redactions = r.redact(line)
        assert "AAAAC3NzaC1lZDI1NTE5" not in text
        # Should also catch github.com as FQDN
        assert "github.com" not in text


class TestPrivateKeyDetection:
    """Test PEM private key block detection."""

    def test_rsa_private_key(self):
        r = Redactor()
        key_block = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIBogIBAAJBALRiMLAHXXXXXXXXXX\n"
            "-----END RSA PRIVATE KEY-----"
        )
        text, redactions = r.redact(f"Found key:\n{key_block}")
        assert "BEGIN RSA PRIVATE KEY" not in text
        assert "[PRIVKEY_1]" in text

    def test_generic_private_key(self):
        r = Redactor()
        key_block = (
            "-----BEGIN PRIVATE KEY-----\n"
            "MIGHAgEAMBMGByqGSM49XXXXXXXX\n"
            "-----END PRIVATE KEY-----"
        )
        text, _ = r.redact(key_block)
        assert "BEGIN PRIVATE KEY" not in text

    def test_openssh_private_key(self):
        r = Redactor()
        key_block = (
            "-----BEGIN OPENSSH PRIVATE KEY-----\n"
            "b3BlbnNzaC1rZXktdXXXXXXXX\n"
            "-----END OPENSSH PRIVATE KEY-----"
        )
        text, _ = r.redact(key_block)
        assert "OPENSSH" not in text


class TestJWTDetection:
    """Test JWT token detection."""

    def test_standard_jwt(self):
        r = Redactor()
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        text, redactions = r.redact(f"Token: {jwt}")
        assert jwt not in text
        assert "[JWT_1]" in text

    def test_jwt_in_curl(self):
        r = Redactor()
        cmd = 'curl -H "Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5c.eyJpc3MiOiJodHRwczovL2FjY291b.SflKxwRJSMeKKF2QT4fwpMe"'
        text, _ = r.redact(cmd)
        assert "eyJhbGci" not in text


class TestBearerTokenDetection:
    """Test bearer token in auth header detection."""

    def test_bearer_header(self):
        r = Redactor()
        text, redactions = r.redact(
            "Authorization: Bearer sk-proj-abcdefghijklmnopqrstuvwxyz123456"
        )
        assert "sk-proj-abcdefghijklmnopqrstuvwxyz123456" not in text

    def test_bearer_prefix(self):
        r = Redactor()
        text, _ = r.redact("Bearer some_long_opaque_token_value_here_1234")
        assert "some_long_opaque_token_value_here_1234" not in text


class TestSSHFingerprintDetection:
    """Test SSH key fingerprint detection."""

    def test_sha256_fingerprint(self):
        r = Redactor()
        fp = "SHA256:nThbg6kXUpJWGl7E1IGOCspRomTxdCARLviKw6E5SY8"
        text, redactions = r.redact(f"Host key fingerprint is {fp}")
        assert fp not in text
        assert "[SSHFP_1]" in text

    def test_md5_fingerprint(self):
        r = Redactor()
        fp = "MD5:16:27:ac:a5:76:28:2d:36:63:1b:56:4d:eb:df:a6:48"
        text, _ = r.redact(f"Fingerprint: {fp}")
        assert fp not in text


class TestCertFingerprintDetection:
    """Test certificate fingerprint detection."""

    def test_sha256_cert(self):
        r = Redactor()
        fp = "SHA256:AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99"
        text, _ = r.redact(f"Cert fingerprint: {fp}")
        assert "AA:BB:CC:DD" not in text


class TestGitRemoteDetection:
    """Test git remote URL detection."""

    def test_ssh_remote(self):
        r = Redactor()
        remote = "git@github.com:user/private-repo.git"
        text, redactions = r.redact(f"Clone: {remote}")
        assert remote not in text
        assert "[GITREMOTE_1]" in text

    def test_ssh_remote_no_git_suffix(self):
        r = Redactor()
        text, _ = r.redact("origin git@gitlab.corp.io:team/service")
        assert "git@gitlab.corp.io:team/service" not in text


class TestUserAtHostDetection:
    """Test user@host SSH-style pattern detection."""

    def test_ssh_login(self):
        r = Redactor()
        text, redactions = r.redact("ssh root@prod-server")
        login_r = [rd for rd in redactions if rd.category == "user_at_host"]
        assert len(login_r) >= 1

    def test_scp_target(self):
        r = Redactor()
        text, _ = r.redact("scp file.txt deploy@staging-box:/opt/app")
        assert "deploy@staging-box" not in text


class TestEnvSecretDetection:
    """Test environment variable secret detection."""

    def test_password_assignment(self):
        r = Redactor()
        text, redactions = r.redact("export PASSWORD=s3cret_p@ss!")
        assert "PASSWORD=s3cret_p@ss!" not in text
        assert "[ENV_1]" in text

    def test_database_url(self):
        r = Redactor()
        text, _ = r.redact("DATABASE_URL=postgres://admin:pw@db:5432/prod")
        assert "DATABASE_URL=postgres" not in text

    def test_aws_secret_key(self):
        r = Redactor()
        text, _ = r.redact("AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
        assert "wJalrXUtnFEMI" not in text


class TestDockerImageDetection:
    """Test Docker image reference detection."""

    def test_registry_image(self):
        r = Redactor()
        text, redactions = r.redact("Pull: registry.corp.io/team/myapp:v2.1.0")
        assert "registry.corp.io/team/myapp" not in text

    def test_ecr_image(self):
        r = Redactor()
        text, _ = r.redact("Image: 123456789.dkr.ecr.us-east-1.amazonaws.com/myapp:latest")
        assert "123456789.dkr.ecr" not in text


class TestK8sPodDetection:
    """Test Kubernetes pod name detection."""

    def test_typical_pod_name(self):
        r = Redactor()
        text, redactions = r.redact("Pod myapp-deploy-7d8f9c6b4f-x2k9j is CrashLooping")
        pod_r = [rd for rd in redactions if rd.category == "k8s_pod"]
        assert len(pod_r) >= 1

    def test_complex_pod_name(self):
        r = Redactor()
        text, _ = r.redact("kubectl logs api-gateway-5b8c9d7f44-m3kp2")
        assert "api-gateway-5b8c9d7f44-m3kp2" not in text


class TestHexStringDetection:
    """Test long hex string detection (git SHAs, container IDs)."""

    def test_git_commit_sha(self):
        r = Redactor()
        sha = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"
        text, redactions = r.redact(f"Commit: {sha}")
        assert sha not in text

    def test_docker_container_id(self):
        r = Redactor()
        cid = "a1b2c3d4e5f6"
        text, _ = r.redact(f"Container {cid} is running")
        assert cid not in text

    def test_short_hex_not_redacted(self):
        """Hex strings under 12 chars should not be caught."""
        r = Redactor()
        text, redactions = r.redact("Error code: abc123")
        hex_r = [rd for rd in redactions if rd.category == "hex_string"]
        assert len(hex_r) == 0


# ── Platform Token Tests (sourced from detect-secrets) ──────────


class TestPlatformTokenDetection:
    """Test platform-specific API key/token patterns."""

    def test_github_token(self):
        r = Redactor()
        token = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl"
        text, redactions = r.redact(f"GITHUB_TOKEN={token}")
        assert token not in text
        tok_r = [rd for rd in redactions if rd.category == "github_token"]
        assert len(tok_r) == 1

    def test_gitlab_token(self):
        r = Redactor()
        token = "glpat-xYzAbCdEfGhIjKlMnOpQ"
        text, _ = r.redact(f"export GITLAB_TOKEN={token}")
        assert token not in text

    def test_slack_bot_token(self):
        r = Redactor()
        token = "xoxb-0000000000000-0000000000000-fakefakefakefakefakefake"
        text, _ = r.redact(f"SLACK_TOKEN={token}")
        assert token not in text

    def test_slack_webhook(self):
        r = Redactor()
        url = "https://hooks.slack.com/services/TFAKETEST/BFAKETEST/FAKEFAKEFAKEFAKEFAKETEST"
        text, _ = r.redact(f"webhook: {url}")
        assert url not in text

    def test_stripe_key(self):
        r = Redactor()
        key = "sk_live_FAKEFAKEFAKEFAKEFAKEFAKE"
        text, _ = r.redact(f"STRIPE_KEY={key}")
        assert key not in text

    def test_sendgrid_key(self):
        r = Redactor()
        key = "SG." + "a" * 22 + "." + "B" * 43
        text, _ = r.redact(f"SENDGRID_API_KEY={key}")
        assert key not in text

    def test_twilio_account_sid(self):
        r = Redactor()
        sid = "AC" + "a" * 32
        text, _ = r.redact(f"TWILIO_SID={sid}")
        assert sid not in text

    def test_aws_access_key_id(self):
        r = Redactor()
        key = "AKIAIOSFODNN7EXAMPLE"
        text, _ = r.redact(f"AWS_ACCESS_KEY_ID={key}")
        assert key not in text

    def test_normal_text_not_flagged(self):
        """Platform patterns should not false-positive on normal text."""
        r = Redactor()
        text, redactions = r.redact("The quick brown fox jumps over the lazy dog")
        plat_r = [rd for rd in redactions if rd.category in (
            "github_token", "gitlab_token", "slack_token", "stripe_key",
            "sendgrid_key", "twilio_key", "openai_key", "discord_token",
        )]
        assert len(plat_r) == 0


# ── Entropy Detection Tests ─────────────────────────────────────


class TestEntropyDetection:
    """Test entropy-based secret detection."""

    def test_high_entropy_hex_token(self):
        """Random mixed-charset strings of sufficient length should be flagged."""
        r = Redactor(ner_enabled=False)
        # Not pure hex so regex hex_string won't catch it, but high entropy
        token = "a3F7b2C9d1E8f4A6b0C5d2E7f3A9b1Z8"  # 32-char mixed with Z
        text, redactions = r.redact(f"My token is {token}")
        assert token not in text
        entropy_r = [rd for rd in redactions if rd.category == "high_entropy"]
        assert len(entropy_r) >= 1

    def test_high_entropy_base64_token(self):
        """Random base64 strings should be flagged."""
        r = Redactor(ner_enabled=False)
        token = "aB3xZ9pQ2mK7wR4nL8vY1cF6hJ5tG0dS"  # 32-char mixed
        text, redactions = r.redact(f"Secret: {token}")
        assert token not in text
        entropy_r = [rd for rd in redactions if rd.category == "high_entropy"]
        assert len(entropy_r) >= 1

    def test_low_entropy_not_flagged(self):
        """Repetitive strings should NOT be flagged."""
        r = Redactor(ner_enabled=False)
        token = "aaaaaaaaaaaaaaaa"  # 16 chars but zero entropy
        text, redactions = r.redact(f"Value: {token}")
        entropy_r = [rd for rd in redactions if rd.category == "high_entropy"]
        assert len(entropy_r) == 0

    def test_short_tokens_not_flagged(self):
        """Tokens under 16 chars should NOT be flagged by entropy."""
        r = Redactor(ner_enabled=False)
        text, redactions = r.redact("Code: abc123")
        entropy_r = [rd for rd in redactions if rd.category == "high_entropy"]
        assert len(entropy_r) == 0

    def test_normal_words_not_flagged(self):
        """Normal English words should NOT be flagged."""
        r = Redactor(ner_enabled=False)
        text, redactions = r.redact(
            "The implementation of this algorithm is straightforward"
        )
        entropy_r = [rd for rd in redactions if rd.category == "high_entropy"]
        assert len(entropy_r) == 0

    def test_urls_not_flagged_by_entropy(self):
        """URLs should be caught by regex, not entropy."""
        r = Redactor(ner_enabled=False)
        text, redactions = r.redact(
            "Visit https://example.com/api/v1/users/list"
        )
        entropy_r = [rd for rd in redactions if rd.category == "high_entropy"]
        assert len(entropy_r) == 0

    def test_entropy_consistent_placeholder(self):
        """Same high-entropy token should get same placeholder across calls."""
        r = Redactor(ner_enabled=False)
        token = "xK9mP3qR7vL2nT5wA8bC4dF6gH1jE0s"
        text1, _ = r.redact(f"First: {token}")
        text2, _ = r.redact(f"Again: {token}")
        assert "[SECRET_1]" in text1
        assert "[SECRET_1]" in text2

    def test_entropy_unredact(self):
        """Entropy redactions should be reversible."""
        r = Redactor(ner_enabled=False)
        token = "zY8xW7vU6tS5rQ4pO3nM2lK1jI0hG9f"
        text, _ = r.redact(f"Key: {token}")
        assert token not in text
        restored = r.unredact(text)
        assert token in restored


class TestEntropyNoFalsePositives:
    """Ensure entropy detection doesn't catch common programming constructs."""

    def test_python_code_not_flagged(self):
        r = Redactor(ner_enabled=False)
        text, redactions = r.redact(
            "def calculate_total_price(items, discount_rate):"
        )
        entropy_r = [rd for rd in redactions if rd.category == "high_entropy"]
        assert len(entropy_r) == 0

    def test_json_structure_not_flagged(self):
        r = Redactor(ner_enabled=False)
        text, redactions = r.redact(
            '{"username": "admin", "role": "administrator"}'
        )
        entropy_r = [rd for rd in redactions if rd.category == "high_entropy"]
        assert len(entropy_r) == 0


# ── NER Detection Tests ─────────────────────────────────────────


class TestNERDetection:
    """Test Presidio NER-based entity detection. Skipped if not installed."""

    @pytest.fixture(autouse=True)
    def _require_presidio(self):
        pytest.importorskip("presidio_analyzer")

    def test_person_name_detected(self):
        r = Redactor(ner_enabled=True)
        text, redactions = r.redact("Please contact John Smith for details")
        ner_r = [rd for rd in redactions if rd.category == "ner_person"]
        # Presidio should detect "John Smith"
        assert len(ner_r) >= 1
        assert "John Smith" not in text

    def test_location_detected(self):
        r = Redactor(ner_enabled=True)
        text, redactions = r.redact(
            "Our office moved to New York last month"
        )
        ner_r = [rd for rd in redactions if rd.category == "ner_location"]
        assert len(ner_r) >= 1

    def test_organization_detected(self):
        r = Redactor(ner_enabled=True)
        text, redactions = r.redact(
            "We signed a contract with Microsoft Corporation"
        )
        ner_r = [rd for rd in redactions if rd.category == "ner_organization"]
        assert len(ner_r) >= 1

    def test_phone_number_detected(self):
        r = Redactor(ner_enabled=True)
        text, redactions = r.redact("Call me at 555-867-5309")
        ner_r = [rd for rd in redactions if rd.category == "ner_phone_number"]
        assert len(ner_r) >= 1

    def test_ner_unredact(self):
        r = Redactor(ner_enabled=True)
        text, _ = r.redact("Send it to John Smith tomorrow")
        assert "John Smith" not in text
        restored = r.unredact(text)
        assert "John Smith" in restored

    def test_ner_skips_already_redacted(self):
        """NER should not double-redact values already caught by regex."""
        r = Redactor(ner_enabled=True)
        text, redactions = r.redact(
            "Email josh@company.com about this issue"
        )
        # Email should be caught by regex, not duplicated by NER
        email_r = [rd for rd in redactions if "josh@company.com" == rd.original]
        assert len(email_r) == 1


class TestNERDisabled:
    """Test that NER is cleanly skipped when disabled."""

    def test_no_ner_when_disabled(self):
        r = Redactor(ner_enabled=False)
        text, redactions = r.redact("Please contact John Smith for details")
        ner_r = [rd for rd in redactions if rd.category.startswith("ner_")]
        assert len(ner_r) == 0

    def test_regex_still_works_when_ner_disabled(self):
        r = Redactor(ner_enabled=False)
        text, redactions = r.redact("Server 192.168.1.50 on port 22")
        assert "192.168.1.50" not in text
        assert "[IP_1]" in text

