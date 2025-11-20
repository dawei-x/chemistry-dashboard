from app import db
from datetime import datetime

class LLMMetrics(db.Model):
    __tablename__ = 'llm_metrics'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    session_device_id = db.Column(db.Integer, db.ForeignKey('session_device.id', ondelete='CASCADE'), nullable=False, unique=True)
    
    # Scores (0-100 scale, matching transcript table)
    emotional_tone_score = db.Column(db.Integer)
    analytic_thinking_score = db.Column(db.Integer)
    clout_score = db.Column(db.Integer)
    authenticity_score = db.Column(db.Integer)
    certainty_score = db.Column(db.Integer)
    
    # Explanations for each score
    emotional_tone_explanation = db.Column(db.Text)
    analytic_thinking_explanation = db.Column(db.Text)
    clout_explanation = db.Column(db.Text)
    authenticity_explanation = db.Column(db.Text)
    certainty_explanation = db.Column(db.Text)
    
    # Metadata
    llm_model = db.Column(db.String(50), default='gpt-4o')
    transcript_count = db.Column(db.Integer)  # Number of transcripts analyzed
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    session_device = db.relationship("SessionDevice", backref="llm_metrics", uselist=False)

    def __init__(self, session_device_id, emotional_tone_score, analytic_thinking_score, 
                 clout_score, authenticity_score, certainty_score,
                 emotional_tone_explanation, analytic_thinking_explanation,
                 clout_explanation, authenticity_explanation, certainty_explanation,
                 transcript_count, llm_model='gpt-4o'):
        self.session_device_id = session_device_id
        self.emotional_tone_score = emotional_tone_score
        self.analytic_thinking_score = analytic_thinking_score
        self.clout_score = clout_score
        self.authenticity_score = authenticity_score
        self.certainty_score = certainty_score
        self.emotional_tone_explanation = emotional_tone_explanation
        self.analytic_thinking_explanation = analytic_thinking_explanation
        self.clout_explanation = clout_explanation
        self.authenticity_explanation = authenticity_explanation
        self.certainty_explanation = certainty_explanation
        self.transcript_count = transcript_count
        self.llm_model = llm_model

    def json(self):
        return dict(
            id=self.id,
            session_device_id=self.session_device_id,
            emotional_tone_score=self.emotional_tone_score,
            analytic_thinking_score=self.analytic_thinking_score,
            clout_score=self.clout_score,
            authenticity_score=self.authenticity_score,
            certainty_score=self.certainty_score,
            emotional_tone_explanation=self.emotional_tone_explanation,
            analytic_thinking_explanation=self.analytic_thinking_explanation,
            clout_explanation=self.clout_explanation,
            authenticity_explanation=self.authenticity_explanation,
            certainty_explanation=self.certainty_explanation,
            transcript_count=self.transcript_count,
            llm_model=self.llm_model,
            created_at=self.created_at.isoformat() if self.created_at else None,
            updated_at=self.updated_at.isoformat() if self.updated_at else None
        )

    @staticmethod
    def verify_fields():
        # No special validation needed for this table
        return True, None