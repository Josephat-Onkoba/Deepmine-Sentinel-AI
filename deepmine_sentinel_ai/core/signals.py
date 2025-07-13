# core/signals.py

from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Stope, StopeProfile
from .utils import profile_stope

@receiver(post_save, sender=Stope)
def generate_profile_summary(sender, instance, created, **kwargs):
    if created:
        features = {
            'rqd': instance.rqd,
            'hr': instance.hr,
            'depth': instance.depth,
            'dip': instance.dip,
            'direction': instance.direction,
            'Undercut_wdt': instance.undercut_wdt,
            'rock_type': instance.rock_type,
            'support_type': instance.support_type,
            'support_density': instance.support_density,
            'support_installed': instance.support_installed,
        }

        summary = profile_stope(features)

        StopeProfile.objects.create(stope=instance, summary=summary)
