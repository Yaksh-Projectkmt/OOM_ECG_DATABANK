def user_has_feature(user, code):
    """
    Checks if the user's current subscription includes a specific feature.
    """
    try:
        subscription = user.usersubscription
        return subscription.plan.features.filter(code=code).exists()
    except:
        return False


from subscription.models import Plan, UserSubscription
from django.contrib.auth import get_user_model
from django.utils import timezone
from datetime import datetime
from subscription.models import DownloadPrice

def sync_subscription_from_mongo(user):
    from authuser.views import users_collection
    from authuser.views import handle_plan_expiry
    mongo_user = users_collection.find_one({"username": user.username})
    handle_plan_expiry(mongo_user)

    if not mongo_user:
        return

    # Get package from MongoDB
    package = mongo_user.get("package")
    if not package:
        return

    # Normalize plan name
    package_name = str(package).strip().lower()

    # Find matching Django plan
    try:
        plan = Plan.objects.get(name__iexact=package_name)
    except Plan.DoesNotExist:
        return

    # Get or create user's subscription
    sub, created = UserSubscription.objects.get_or_create(user=user)

    # Update plan if changed
    if sub.plan_id != plan.id:
        sub.plan = plan

    # Sync expiry date if exists in MongoDB
    expiry = mongo_user.get("expiry")  # must be YYYY-MM-DD in MongoDB
    if expiry:
        try:
            # Convert string to date
            sub.end_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        except Exception:
            pass  # ignore incorrect formats

    sub.save()
    return sub
from subscription.models import DownloadPrice
from authuser.models import Wallet

def get_download_price(user, file_type):
    try:
        return DownloadPrice.objects.get(
            role=user.role,
            file_type=file_type
        ).price
    except DownloadPrice.DoesNotExist:
        return 0  # Default 0 if admin did not configure price


def create_download_history(user, file_type, price, wallet_before, wallet_after, status):
    from authuser.views import download_history_collection
    from authuser.views import users_collection
    download_history_collection.insert_one({
        "email": user.email,
        "file_type": file_type,
        "price": price,
        "wallet_before": wallet_before,
        "wallet_after": wallet_after,
        "status": status,
        "download_at": timezone.now()
    })