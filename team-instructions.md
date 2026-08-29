# Team Collaboration Guidelines & Instructions

Welcome to the repository for our final year mega project! To ensure our development process runs smoothly and our codebase remains clean and conflict-free, every team member must follow the workflow outlined below.

## 1. Getting Started

Before you start building, you need to get the project running locally on your machine.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/samarjeets10/<our-repo-name>.git
   ```
   *(Note: Replace `<our-repo-name>` with the actual repository name.)*

2. **Navigate into the project directory:**
   ```bash
   cd <our-repo-name>
   ```

3. **Install dependencies:** 
   *(Assuming our standard frontend setup with React/Vite)*
   ```bash
   npm install
   ```

4. **Start the local development server:**
   ```bash
   npm run dev
   ```

## 2. The Golden Rule: Branching
**NEVER commit or push directly to the `main` branch.** 

The `main` branch must always contain working, production-ready code. Whenever you start a new task, you must create a separate branch.

**Branch Naming Conventions:**
Use descriptive prefixes so everyone knows what you are working on:
* `feature/` for new additions (e.g., `feature/navbar-ui`, `feature/auth-flow`)
* `bugfix/` for fixing issues (e.g., `bugfix/login-crash`)
* `ui/` for frontend layout tweaks (e.g., `ui/feed-card-styling`)

## 3. Daily Development Workflow

Follow these exact steps every time you pick up a new task:

1. **Sync your local repository:** Always start by making sure you have the latest code from the team.
   ```bash
   git checkout main
   git pull origin main
   ```
2. **Create your feature branch:**
   ```bash
   git checkout -b feature/your-descriptive-branch-name
   ```
3. **Build and Commit:** Write your code, create your components, and commit your work frequently. Write clear, meaningful commit messages.
   ```bash
   git add .
   git commit -m "Added responsive layout to the user dashboard"
   ```
4. **Push your branch to GitHub:**
   ```bash
   git push origin feature/your-descriptive-branch-name
   ```

## 4. Creating a Pull Request (PR)

Once your feature is complete and tested locally, it's time to merge it into the main project.

1. Go to our repository page on GitHub.
2. You will see a green banner that says **Compare & pull request** next to your recently pushed branch. Click it.
3. **Title and Description:** Give your PR a clear title. In the description, briefly explain what changes you made and note any specific UI/UX elements or logic your teammates should look out for.
4. **Request a Review:** Add at least one team member as a reviewer. 
5. **DO NOT MERGE YOUR OWN PR.** Wait for a teammate to review your code, approve it, and handle the merge. This is crucial for catching errors before they hit `main`.

## 5. Handling Merge Conflicts

If you and a teammate edit the exact same line in the same file, Git won't know which version to keep, resulting in a merge conflict.

* **Prevent them:** Communicate with the team about which files you are working on. Pull from `main` frequently.
* **Resolve them:** If a conflict happens when you try to merge your PR, GitHub will alert you. You can often resolve these directly in GitHub's web editor or by pulling the `main` branch into your local feature branch (`git pull origin main`), fixing the highlighted conflicts in your code editor (like VS Code), committing the fix, and pushing again.

## 6. Coding Best Practices
* **Keep PRs small:** Don't build the entire app in one branch. Break tasks down (e.g., do the header in one branch, the footer in another). Smaller PRs are easier to review.
* **Format your code:** Keep your code clean, properly indented, and remove unused console logs before pushing.
* **Communicate:** If you are stuck or planning to restructure a major component, discuss it with the team first!