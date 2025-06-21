import redis

def test_redis_set_get(start_services):
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.set('ping','pong')
    assert r.get('ping') == b'pong'
