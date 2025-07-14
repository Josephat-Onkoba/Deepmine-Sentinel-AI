# core/signals.py
# Updated for impact-based prediction system

from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from .models import Stope, OperationalEvent, ImpactScore, ImpactHistory

@receiver(post_save, sender=Stope)
def create_impact_score(sender, instance, created, **kwargs):
    """
    Automatically create an ImpactScore record when a new Stope is created.
    Update baseline when Stope is modified.
    """
    if created:
        # Create initial impact score for new stope
        ImpactScore.objects.create(
            stope=instance,
            current_score=instance.baseline_stability_score,
            baseline_component=instance.baseline_stability_score,
            operational_component=0.0,
            temporal_component=0.0,
            monitoring_component=0.0,
            last_calculation_method='automatic'
        )
    else:
        # Update existing impact score when stope is modified
        try:
            impact_score = instance.impact_score
            old_score = impact_score.current_score
            impact_score.save()  # This will trigger recalculation
            
            # Log the change in history if significant
            if abs(impact_score.current_score - old_score) >= 0.05:
                ImpactHistory.objects.create(
                    stope=instance,
                    previous_score=old_score,
                    new_score=impact_score.current_score,
                    previous_risk_level=impact_score.get_risk_level_from_score(old_score),
                    new_risk_level=impact_score.risk_level,
                    change_type='system_recalculation',
                    change_reason=f'Stope geological parameters updated',
                    calculation_method='automatic',
                    calculated_by='system'
                )
        except ImpactScore.DoesNotExist:
            # Create impact score if it doesn't exist
            ImpactScore.objects.create(
                stope=instance,
                current_score=instance.baseline_stability_score,
                baseline_component=instance.baseline_stability_score,
                last_calculation_method='automatic'
            )


@receiver(post_save, sender=OperationalEvent)
def update_impact_score_on_event(sender, instance, created, **kwargs):
    """
    Update the stope's impact score when an operational event is recorded.
    """
    if created:
        try:
            impact_score = instance.stope.impact_score
            old_score = impact_score.current_score
            old_risk_level = impact_score.risk_level
            
            # Recalculate the impact score
            impact_score.save()  # This triggers recalculation
            
            # Create history record for this change
            ImpactHistory.objects.create(
                stope=instance.stope,
                previous_score=old_score,
                new_score=impact_score.current_score,
                previous_risk_level=old_risk_level,
                new_risk_level=impact_score.risk_level,
                change_type='event_impact',
                change_reason=f'{instance.get_event_type_display()} event: {instance.description[:100]}',
                related_operational_event=instance,
                calculation_method='automatic',
                calculated_by='system'
            )
            
        except ImpactScore.DoesNotExist:
            # Create impact score if it doesn't exist
            ImpactScore.objects.create(
                stope=instance.stope,
                current_score=instance.stope.baseline_stability_score,
                baseline_component=instance.stope.baseline_stability_score,
                last_calculation_method='automatic'
            )


# ===== OPTIMIZED SIGNALS FOR IMPACT-BASED SYSTEM =====
# Automatic impact score management and history tracking
