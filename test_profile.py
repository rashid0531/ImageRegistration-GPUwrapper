
try:
    import line_profiler
except ImportError:
    print("No line profiler, skipping test.")
    import sys
    sys.exit(0)


def assert_stats(profile, name):
    profile.print_stats()
    stats = profile.get_stats()
    assert len(stats.timings) > 0, "No profile stats."
    for key, timings in stats.timings.items():
        if key[-1] == name:
            assert len(timings) > 0
            break
    else:
        raise ValueError("No stats for %s." % name)


from GpuWrapper import cudaWarpPerspectiveWrapper
func = cudaWarpPerspectiveWrapper
profile = line_profiler.LineProfiler(func)
profile.runcall(func, 19)
assert_stats(profile, func.__name__)

from GpuWrapper import match_feature
func = match_feature
profile = line_profiler.LineProfiler(func)
profile.runcall(func, 19)
assert_stats(profile, func.__name__)
