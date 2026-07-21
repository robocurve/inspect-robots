# 0025: Robocurve-style landing page and quickstart refresh

## Goal

Two user-visible changes to inspectrobots.org, no framework code touched:

1. `docs/guide/quickstart.md` catches up with the README rewrite (#147): the
   real-rig setup wizard, LLM-as-policy (`--policy agent`), CaP-X, `--sim`,
   and the `view`/`video` commands are absent from the page today.
2. The landing page (`website/src/pages/index.tsx` + CSS) adopts the clean
   editorial style of robocurve.org, the parent org's site.

## Non-goals

- No changes to guide pages other than quickstart and the `cli.md`
  carve-outs listed in Changes §5.
- No changes to the API generator, CI, or URL structure.
- No content removal from the landing page: same sections, restyled.
- Dark mode stays supported; it follows the existing dark palette, not a
  dark variant of the robocurve cream.

## Source of truth for the style

Extracted from robocurve.org's shipped CSS (site.TKPXBSW6.css):

- Fonts: Space Grotesk (display + body), IBM Plex Mono (eyebrows/data).
- Page surface `#FFFAED`; cards are white (`#ffffff`) with 1px hairline
  border `#e6ddcd`; radius 6-10px; shadows absent or near-invisible.
- Text: strong `#1a1511`, body `#403830`, muted `#736853`. Headings are
  weight 500-600 with letter-spacing -0.01em to -0.03em, near-black (not
  teal).
- Accent: `#005f73` (identical to our existing primary), hover `#024e5e`.
  Used only for buttons, links, eyebrows, small marks.
- Eyebrow pattern: mono, uppercase, 0.16em tracking, accent color.
- Hero: text left on bare cream (no band, no gradient), illustration right,
  one filled accent button + one hairline outline button.

## Changes

### 1. Fonts (self-hosted, no external requests)

- `npm install @fontsource/space-grotesk @fontsource/ibm-plex-mono` in
  `website/` (lockfile updated).
- `custom.css` imports weights 400/500/600/700 (Space Grotesk) and 400/500
  (IBM Plex Mono), then sets `--ifm-font-family-base` to Space Grotesk and
  keeps the default code font for code blocks (IBM Plex Mono is for
  eyebrow labels, not code; Infima's monospace stack stays).

### 2. Sitewide theme (`website/src/css/custom.css`), light mode only

Scoping rule: every light-mode value goes under `[data-theme='light']`,
never bare `:root`. Infima defines several of these variables only on
`:root` (`--ifm-font-color-base`, `--ifm-heading-color`,
`--ifm-navbar-background-color`) and implements dark mode by swapping
*other* variables, so a bare-`:root` override in custom.css (loaded after
Infima) would leak near-black text and a cream navbar into dark mode.
Docusaurus 3 always stamps `data-theme` on `<html>`, so the selector is
reliable.

- `[data-theme='light']`: `--ifm-background-color: #FFFAED`,
  `--ifm-background-surface-color: #ffffff`,
  `--ifm-font-color-base: #403830`, `--ifm-heading-color: #1a1511`,
  navbar background cream.
- Heading weight drops from 700 to 600 (mode-independent, fine on
  `:root`); add the negative tracking via an `h1-h4` rule.
- Navbar: keep the existing `--ifm-navbar-shadow` faux-hairline mechanism
  (a real `border-bottom` would add 1px to the fixed navbar's box and
  nudge anchor-scroll offsets) but recolor it to `0 1px 0 #e6ddcd` in
  light mode; dark mode keeps the current value.
- Footer: keep `#003f4d` dark footer (robocurve also closes on an ink
  band); unchanged.
- Dark mode block: untouched except anything needed to keep contrast after
  the font swap (no color changes expected).
- Docs pages inherit the cream background and new type automatically;
  admonitions/code blocks keep their Infima surfaces (white/gray) which
  reads as "cards on cream", consistent with the target style.

### 3. Landing page restyle (`index.module.css`, minor `index.tsx` edits)

Same sections, same copy (except where noted), new skin:

- Hero: drop the gradient band and logo drop-shadow. Bare cream, two-column
  grid: text left (H1 near-black ~clamp(2.5rem,6vw,3.75rem), weight 600,
  -0.03em; tagline becomes body-size muted lead, not teal bold), logo right
  at a calmer size. Buttons: filled teal `Get started` (radius 6px) +
  hairline outline `Concepts` (border `#d2c6b2`, text near-black), matching
  robocurve's button pair.
- `cardLabel` (THE VLA BRAIN / THE ROBOT OR SIM) becomes the robocurve
  eyebrow: IBM Plex Mono, uppercase, 0.16em tracking, teal, on **white**
  cards with hairline borders; H3 near-black; body text `#403830`. The
  dark-teal gradient fill, big radius, and 45px shadows go away.
- `creamSection` alternation disappears (whole page is cream); section
  separation via spacing and, where needed, a hairline `border-top`.
- Feature cards and plugin cards: white surface, hairline border, 10px
  radius, no translateY hover; hover = border-color to teal. Plugin strip
  goes from 5 skinny columns to exactly 3 (3+2 rows; 2 columns would
  orphan the fifth card), collapsing to 1 on mobile as today.
- Section H2s: near-black, weight 600, -0.02em tracking, robocurve's
  section size (~clamp(1.6rem,4vw,2.125rem)); optional mono eyebrows above
  key sections ("THE FRAMEWORK", "QUICKSTART", ...) only if they read
  naturally; skip if forced.
- Inspect AI mapping table: hairline borders, white header row; override
  `--ifm-table-stripe-background` (Infima's default gray zebra) so rows
  sit flat on the card surface.
- Dark mode: every literal light value in `index.module.css` (`#ffffff`
  card surfaces, `#e6ddcd` hairlines, near-black text) gets an explicit
  `[data-theme='dark'] .class` override (attribute selectors are not
  hashed by CSS modules, so this composes with module classes). The
  existing scoped `.heroOutlineButton` fix is superseded by the new
  button styles under the same rule. Test both modes by screenshot; this
  exact class of bug (light literal leaking into dark mode) already bit
  once.

### 4. CLI guide carve-outs (`docs/guide/cli.md`)

Two small consistency edits, no restructuring:

- Document the existing `--transcript` flag in the `inspect` section (the
  new quickstart references it; the CLI reference must not lag a flag the
  quickstart teaches).
- The page mixes fictional component names with the real YAM ones its own
  walkthrough uses. Fix every occurrence, not just the first: the generic
  config example (`policy = molmoact2-yam`, `embodiment = yam-bimanual`,
  `sim_embodiment = yam-bimanual-isaac`), the `checkpoint =
  ~/ckpts/molmoact2-yam.pt` `-P` example (implausible for `molmoact2`,
  which is a server client), and the `--sim` run example
  (`--policy molmoact2-yam`). Use the real names (`molmoact2`,
  `yam_arms`) wherever a real counterpart exists; where none does (the
  sim embodiment, the checkpoint illustration), use openly generic
  placeholders or a real parameter so the page never presents a
  half-real name. The result must be internally consistent across the
  whole page.

### 5. Quickstart guide (`docs/guide/quickstart.md`)

Restructured to mirror the README while staying a docs page (links instead
of duplicating deep detail). Section order:

1. **Install** (unchanged).
2. **Run your first evaluation** (mock world Python, unchanged): stays
   first because it is the only zero-hardware path. This block keeps the
   docs variant (separate `print` lines with output comments); the
   verbatim-README rule below applies to the newly added CLI blocks, not
   to this pre-existing one.
3. **Use registry names** (unchanged).
4. **From the command line**: extend the existing block with `view` and
   `video` (with the ffmpeg note), keeping `list`/`run`/`inspect`. Drop
   the section's current intro sentence about the setup wizard: with the
   new "On a real robot" section immediately following, that forward
   reference becomes redundant.
5. **On a real robot** (new): rig plugin install (`inspect-robots-yam`
   example), `inspect-robots setup` wizard writing
   `~/.config/inspect-robots/config.ini`, the policy-server caveat
   (molmoact2 is a client; server start + `curl .../act` check, link to
   the yam plugin README), then `inspect-robots "place the fork on the
   plate"`, the Rerun live-viewer paragraph, flag overrides, and the
   default guardrails sentence (`--disable-guardrails` opt-out). Link to
   the CLI guide for config-file details.
6. **Drive the robot with an LLM** (new): `.env` with `ANTHROPIC_API_KEY`,
   `uv pip install inspect-robots-agent`, the README's run command, plus
   the no-hardware variant `--embodiment cubepick` so readers without a
   rig can run it immediately; transcript reading via
   `inspect logs/... --transcript` and `view`.
7. **Generate robot policy code with CaP-X** (new, short): one paragraph +
   install/run block + link to the plugin README for server bringup and
   the trust boundary.
8. **Run in simulation** (new, short): `--sim`.
9. **Next steps** (unchanged links; add Plugins guide link).

Style rules apply (no em dashes, restrained bold, no decorative emoji).
Code blocks must match the README verbatim where they overlap (the README
was reviewed for correctness in #147; do not invent new flags). The
surrounding prose is paraphrased, not copied: the README's prose contains
em dashes the docs style rules forbid.

## Verification

- `cd website && npm run build` green (strict broken-link/anchor checks)
  and `npm run typecheck` green (`docusaurus build` transpiles TS without
  type-checking, and CI's docs job only builds, so this is the only gate
  that type-checks `index.tsx`).
- Screenshots of `/` and `/guide/quickstart/` in light and dark mode
  (headless Chrome, `--blink-settings=preferredColorScheme=1` for light,
  `=0` for dark; Blink's enum is kDark=0, kLight=1) reviewed
  against robocurve.org for: cream page, near-black headings, hairline
  cards, no gradients, legible dark mode.
- `npm run serve` preview offered to the user before merge (explicit user
  request for this change).
- CI `docs-build` green on the PR.

## Risks

- Font imports from `@fontsource` must resolve through the css-loader
  module resolution (`@import "@fontsource/space-grotesk/400.css";`); if
  the build rejects bare module imports, fall back to importing them via
  `docusaurus.config.ts` client modules.
- Cream background sitewide could reduce code-block contrast in docs;
  verify one guide page screenshot, and if it reads poorly keep docs pages
  on the default background by scoping the cream to the landing page only
  (decision recorded in the PR description).
- The local-search plugin styles inputs against Infima variables; check the
  navbar search field still looks right on cream.
- Space Grotesk ships no italic cuts, so markdown emphasis renders as
  browser-synthesized oblique. Accepted tradeoff (robocurve.org has the
  same property); revisit only if emphasis-heavy pages read badly.
