from django import template
from subscription.utils import user_has_feature

register = template.Library()

@register.filter(name='has_feature')
def has_feature(user, feature_code):
    return user_has_feature(user, feature_code)
