declare module '*.md' { const text: string; export default text; }
declare module '*.py' { const text: string; export default text; }
interface ReviewerEnv { OPENAI_API_KEY: string; GITHUB_WEBHOOK_SECRET: string; }
interface PublisherEnv { GITHUB_PRIVATE_KEY: string; }
