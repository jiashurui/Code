import googlemaps
from datetime import datetime

gmaps = googlemaps.Client(key='AIzaSyAgCU29FYbjgwppooCvPCcVet1EM0n3moE')

# Geocoding an address
geocode_result = gmaps.geocode('1600 Amphitheatre Parkway, Mountain View, CA')

# Look up an address with reverse geocoding
reverse_geocode_result = gmaps.reverse_geocode((37.443182, 138.850909))

# reverse_geocode_result = gmaps.reverse_geocode((37.434572, 138.854363))

print(reverse_geocode_result)

# 伪代码示例
# https://maps.googleapis.com/maps/api/place/nearbysearch/json?location=37.434028, 138.787942&radius=10&type=intersection&key=AIzaSyAgCU29FYbjgwppooCvPCcVet1EM0n3moE
# https://maps.googleapis.com/maps/api/place/nearbysearch/json?location=37.7937%2C138.787942&radius=10&type=intersection&key=AIzaSyAgCU29FYbjgwppooCvPCcVet1EM0n3moE

#
# curl -X POST -d '{
#   "includedTypes": ["intersection"],
#   "locationRestriction": {
#     "circle": {
#       "center": {
#         "latitude": 37.434028,
#         "longitude": 138.787942},
#       "radius": 10.0
#     }
#   }
# }' \
# -H 'Content-Type: application/json' -H "X-Goog-Api-Key: AIzaSyAgCU29FYbjgwppooCvPCcVet1EM0n3moE" \
# -H "X-Goog-FieldMask: places.displayName,places.formattedAddress" \
# https://places.googleapis.com/v1/places:searchNearby
#
#
# places.types
