# Generated by Django 3.0.7 on 2020-07-12 14:49

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('reviews', '0002_review'),
    ]

    operations = [
        migrations.DeleteModel(
            name='Choice',
        ),
        migrations.DeleteModel(
            name='Question',
        ),
    ]