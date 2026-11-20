**Dream plugin (0.1.0):** new `inspect-robots-dream` package registers `dream`,
a policy that runs Dream Labs' hosted Dream-1 on the bimanual YAM layout
(`yam_arms`, `yam_arena`). An instruction Dream refuses as unsafe ends the trial
with the `give_up` terminal action and Dream's stated reason, instead of an
errored trial ([plan 0090](plans/0090-dream-policy-plugin.md)).
