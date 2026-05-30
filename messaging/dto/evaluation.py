from dataclasses import dataclass, field


@dataclass
class Evaluation:
    score: int
    passed: bool
    issues: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    retry_targets: list[str] = field(default_factory=list)
    hard_errors: list[str] = field(default_factory=list)
    summary: str = ""
    skipped: bool = False

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "passed": self.passed,
            "issues": self.issues,
            "strengths": self.strengths,
            "retry_targets": self.retry_targets,
            "hard_errors": self.hard_errors,
            "summary": self.summary,
            "skipped": self.skipped,
        }
