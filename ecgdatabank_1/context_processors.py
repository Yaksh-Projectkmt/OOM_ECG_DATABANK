def user_session(request):
    return {
        'user_session': request.session.get('user_session', None)
    }