from django.contrib.auth.signals import user_logged_in
from django.dispatch import receiver
from subscription.models import UserSubscription, Plan

# @receiver(user_logged_in)
# def sync_plan_on_login(sender, user, request, **kwargs):
#     print("🔔 LOGIN SIGNAL FIRED for:", user.username)

#     # Does the user have a package field?
#     package = getattr(user, "package", None)
#     print("📦 User package =", package)

#     if not package:
#         print("⛔ NO PACKAGE FOUND ON USER OBJECT")
#         return

#     try:
#         plan = Plan.objects.get(name__iexact=package.lower())
#         print("📌 Plan found:", plan.name)
#     except Plan.DoesNotExist:
#         print("⛔ PLAN NOT FOUND IN DJANGO:", package)
#         return

#     sub, created = UserSubscription.objects.get_or_create(user=user)
#     sub.plan = plan
#     sub.save()
#     print("🎉 SYNC DONE:", sub)
