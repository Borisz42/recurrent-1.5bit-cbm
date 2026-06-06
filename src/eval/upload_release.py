import os
import sys
import argparse
import json
import shutil
import subprocess
import urllib.request
import urllib.error

def get_git_remote():
    """Tries to detect the owner/repo from local git configuration."""
    try:
        url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        
        # Parse github.com/owner/repo from various formats:
        # e.g., https://github.com/owner/repo.git or git@github.com:owner/repo.git
        if "github.com" in url:
            parts = url.split("github.com")[-1].lstrip(":/").replace(".git", "").split("/")
            if len(parts) >= 2:
                return parts[0], parts[1]
    except Exception:
        pass
    return None, None

def zip_directory(dir_path, zip_name):
    """Zips a directory if it exists."""
    if not os.path.exists(dir_path):
        print(f"Warning: Directory '{dir_path}' not found. Skipping zipping.")
        return None
    
    zip_file = f"{zip_name}.zip"
    print(f"Creating zip archive for '{dir_path}' -> '{zip_file}'...")
    try:
        shutil.make_archive(zip_name, 'zip', dir_path)
        return zip_file
    except Exception as e:
        print(f"Error zipping '{dir_path}': {e}")
        return None

def github_request(url, token, data=None, method="GET", content_type="application/json"):
    """Performs an API request to GitHub using urllib."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "Python-Release-Uploader",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    
    body = None
    if data is not None:
        if isinstance(data, (dict, list)):
            body = json.dumps(data).encode("utf-8")
            headers["Content-Type"] = content_type
        else:
            body = data
            headers["Content-Type"] = content_type

    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as response:
            res_data = response.read()
            if res_data:
                return json.loads(res_data.decode("utf-8")), response.status
            return None, response.status
    except urllib.error.HTTPError as e:
        err_content = e.read().decode("utf-8")
        try:
            err_json = json.loads(err_content)
            err_msg = err_json.get("message", err_content)
        except Exception:
            err_msg = err_content
        raise RuntimeError(f"GitHub API Error ({e.code}): {err_msg}")
    except Exception as e:
        raise RuntimeError(f"Connection Error: {e}")

def get_or_create_release(owner, repo, tag, token):
    """Gets an existing release or creates a new one."""
    print(f"Checking if release '{tag}' exists...")
    get_url = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
    
    try:
        release, status = github_request(get_url, token, method="GET")
        print(f"Release '{tag}' already exists.")
        return release
    except RuntimeError as e:
        if "Not Found" in str(e):
            print(f"Release '{tag}' not found. Creating a new release...")
            create_url = f"https://api.github.com/repos/{owner}/{repo}/releases"
            payload = {
                "tag_name": tag,
                "name": tag,
                "body": f"Automated training checkpoint release for {tag}.",
                "draft": False,
                "prerelease": False
            }
            release, status = github_request(create_url, token, data=payload, method="POST")
            print(f"Release '{tag}' created successfully.")
            return release
        else:
            raise e

def delete_existing_asset(owner, repo, release_id, asset_name, token):
    """Deletes any asset in the release matching the asset_name to prevent conflicts."""
    assets_url = f"https://api.github.com/repos/{owner}/{repo}/releases/{release_id}/assets"
    assets, status = github_request(assets_url, token, method="GET")
    
    for asset in assets:
        if asset["name"] == asset_name:
            print(f"Found existing asset '{asset_name}'. Deleting to overwrite...")
            del_url = f"https://api.github.com/repos/{owner}/{repo}/releases/assets/{asset['id']}"
            github_request(del_url, token, method="DELETE")
            print(f"Deleted old '{asset_name}'.")
            return

def upload_asset(upload_url_template, file_path, token):
    """Uploads a file to the GitHub Release."""
    filename = os.path.basename(file_path)
    # The upload URL template from github looks like:
    # https://uploads.github.com/repos/owner/repo/releases/id/assets{?name,label}
    # We clean it up and append name query parameter
    base_upload_url = upload_url_template.split("{")[0]
    upload_url = f"{base_upload_url}?name={filename}"
    
    print(f"Uploading '{filename}' ({os.path.getsize(file_path) / 1024 / 1024:.2f} MB)...")
    
    with open(file_path, "rb") as f:
        file_data = f.read()
        
    github_request(
        upload_url,
        token,
        data=file_data,
        method="POST",
        content_type="application/octet-stream"
    )
    print(f"Successfully uploaded '{filename}'!")

def main():
    parser = argparse.ArgumentParser(description="Automate uploading model checkpoints to GitHub Releases.")
    parser.add_argument("--tag", type=str, default="v1.0.0", help="The release tag name (e.g., v1.0.0).")
    parser.add_argument("--repo", type=str, help="Repository path as 'owner/repo' (detected from git if omitted).")
    parser.add_argument("--token", type=str, help="GitHub PAT token (detected from GITHUB_TOKEN env var if omitted).")
    parser.add_argument("--files", nargs="+", help="Explicit list of files/directories to upload. If omitted, default model files are used.")
    
    args = parser.parse_args()
    
    # Resolve token
    token = args.token or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: GitHub Access Token is required.", file=sys.stderr)
        print("Please set the GITHUB_TOKEN environment variable or pass --token <your-token>.", file=sys.stderr)
        sys.exit(1)
        
    # Resolve repository
    owner, repo = None, None
    if args.repo:
        if "/" in args.repo:
            owner, repo = args.repo.split("/", 1)
        else:
            print("Error: --repo must be in the format 'owner/repo'.", file=sys.stderr)
            sys.exit(1)
    else:
        owner, repo = get_git_remote()
        
    if not owner or not repo:
        print("Error: Could not automatically detect GitHub repository owner/name from git remote.", file=sys.stderr)
        print("Please specify the repository using --repo 'owner/repo'.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Target Repository: {owner}/{repo}")
    print(f"Target Tag: {args.tag}")
    
    # Resolve files to upload
    files_to_upload = []
    if args.files:
        files_to_upload = args.files
    else:
        # Default behavior: check local directories and zip them if needed, or upload zip/pt files directly
        candidates = [
            ("adapters", "adapters"),
            ("t_trm_outputs", "t_trm_outputs"),
            ("hybrid_cbm.pt", None)
        ]
        for path, zip_prefix in candidates:
            if os.path.exists(path):
                if os.path.isdir(path) and zip_prefix:
                    # Check if a zip with the same name already exists to avoid re-zipping
                    zip_name = f"{zip_prefix}.zip"
                    if os.path.exists(zip_name):
                        print(f"Using existing zip file '{zip_name}'...")
                        files_to_upload.append(zip_name)
                    else:
                        zip_file = zip_directory(path, zip_prefix)
                        if zip_file:
                            files_to_upload.append(zip_file)
                else:
                    files_to_upload.append(path)
            # Check if zip files exist directly in working directory even if source dirs don't
            elif zip_prefix and os.path.exists(f"{zip_prefix}.zip"):
                files_to_upload.append(f"{zip_prefix}.zip")
                
    if not files_to_upload:
        print("Error: No files found to upload. Please specify files or place default model files in this directory.", file=sys.stderr)
        sys.exit(1)
        
    # Process release and upload assets
    try:
        release = get_or_create_release(owner, repo, args.tag, token)
        release_id = release["id"]
        upload_url = release["upload_url"]
        
        for file_path in files_to_upload:
            if not os.path.exists(file_path):
                print(f"Warning: File '{file_path}' does not exist. Skipping.")
                continue
            
            asset_name = os.path.basename(file_path)
            # Overwrite if exists
            delete_existing_asset(owner, repo, release_id, asset_name, token)
            # Upload
            upload_asset(upload_url, file_path, token)
            
        print("\nAll files uploaded successfully!")
        print(f"View your release at: https://github.com/{owner}/{repo}/releases/tag/{args.tag}")
        
    except Exception as e:
        print(f"\nUpload failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
